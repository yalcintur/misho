from typing import Self, Callable, List, Tuple, TypeVar, Optional, Dict
import random
import re
import time
import asyncio

class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, state: str, parent: Self | None, visit_count: int, action_value: float, value_estimate: float):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = visit_count  # N
        self.action_value = action_value  # Q
        self.value_estimate = value_estimate  # V
        self.is_terminal = self.state.endswith('.') or self.state.endswith('.\n')

    @property
    def is_visited(self) -> bool:
        return self.visit_count > 0

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    def add_children(self, state_values: list[tuple[str, float]]):
        """Add child nodes with their value estimates."""
        self.children.extend([MCTSNode(state=state, parent=self, visit_count=0, 
                                      action_value=value, value_estimate=value) 
                             for state, value in state_values])

    def evaluate_terminal_state(self, question: str) -> float:
        """Evaluate if terminal state solves the 24 game."""
        if not self.is_terminal:
            raise ValueError("Evaluation called on non-terminal state")
        try:
            question_nums = sorted([int(x) for x in question.split()])
            last_line = ''.join(c for c in self.state.split('\n')[-1] 
                              if c in '0123456789+-*/()=')
            parts = last_line.split('=')
            if len(parts) != 2 or parts[1] != '24':
                return 0.0
            
            expr_nums = sorted([int(n) for n in re.findall(r'\d+', parts[0])])
            return 1.0 if expr_nums == question_nums and abs(eval(parts[0]) - 24) <= 1e-6 else 0.0
        except:
            return 0.0

    @property
    def favourite_child(self) -> 'MCTSNode':
        """Return child with highest visit count, breaking ties with Q value."""
        if not self.has_children:
            raise ValueError("Node has no children")
            
        return max(self.children, key=lambda child: (child.visit_count, child.action_value))

class MCTSTree:
    """Single MCTS tree for exploring solutions to a question."""
    
    def __init__(self, root_value: float, question: str, max_expansions: int, 
                 exploration_constant: float, request_queue):
        self.root = MCTSNode(state="", parent=None, visit_count=0, 
                            action_value=0, value_estimate=root_value)
        self.question = question
        self.max_expansions = max_expansions
        self.exploration_constant = exploration_constant
        self.policy_training_data = []
        self.value_training_data = []
        self.request_queue = request_queue
        self.expansion_count = 0
        self.non_terminal_leaves = [self.root]  

    async def get_action_values(self, node: MCTSNode) -> list[tuple[str, float]]:
        """Get action-value pairs from policy-value network."""
        future = asyncio.Future()
        await self.request_queue.put((self.question, node.state, future))
        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            print(f"Error: No response after 60s at {node.state}.")
            return []

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 formula."""
        ucb1 = lambda child: (child.action_value + 
                            self.exploration_constant * (node.visit_count ** 0.5) / 
                            (child.visit_count + 1))
        return max(node.children, key=ucb1)

    def backpropagate(self, node: MCTSNode, value: float):
        """Update node statistics from leaf to root."""
        while node:
            node.visit_count += 1
            node.action_value += (value - node.action_value) / node.visit_count
            node = node.parent

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while (self.expansion_count < self.max_expansions and self.non_terminal_leaves):
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                value = current.evaluate_terminal_state(self.question)
                self.value_training_data.append((current.state, value))
                if value:    
                    self.policy_training_data.append(current.state)
                self.backpropagate(current, value)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            else:
                try:
                    new_states = await self.get_action_values(current)
                    self.non_terminal_leaves.remove(current)
                    current.add_children(new_states)
                    for child in current.children:
                        if not child.is_terminal:
                            self.non_terminal_leaves.append(child)
                    self.expansion_count += 1
                except Exception as e:
                    print(f"Expansion error: {e}")
                    break
            await asyncio.sleep(0)
        return self.policy_training_data, self.value_training_data

class MCTSForest:
    """Forest of MCTS trees for parallel exploration of multiple questions."""
    
    def __init__(self, initial_values: list[float], questions: list[str],
                 max_expansions: int, num_trees: int, 
                 exploration_constant: float, policy_value_fn: Callable,
                 process_policy_trajectory: Callable,
                 process_value_trajectory: Callable,
                 batch_size: int, batch_interval: float):
        # Initialize forest parameters
        self.initial_values = initial_values
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.exploration_constant = exploration_constant
        
        # Set network functions
        self.policy_value_fn = policy_value_fn
        self.process_policy_trajectory = process_policy_trajectory
        self.process_value_trajectory = process_value_trajectory
        
        # Initialize data collection
        self.policy_training_data = []
        self.value_training_data = []
        self.policy_data_counts = {q: 0 for q in questions}
        self.value_data_counts = {q: 0 for q in questions}
        
        # Set up batch processing
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        
        # Initialize active questions tracking
        self.active_questions = set()
        self.active_questions_lock = asyncio.Lock()
        
        # Track API usage and runtime
        self.total_api_calls = 0
        self.start_time = time.time()
        
        # Create initial trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree]:
        """Initialize the forest with trees for each spot."""
        trees = []
        for i in range(self.num_trees):
            question_idx = i % len(self.questions)
            question = self.questions[question_idx]
            self.active_questions.add(question)
            trees.append(self._create_tree(question_idx, question))
        return trees

    def _create_tree(self, question_idx: int, question: str) -> MCTSTree:
        """Create a new MCTS tree."""
        return MCTSTree(
            root_value=self.initial_values[question_idx],
            question=question,
            max_expansions=self.max_expansions,
            exploration_constant=self.exploration_constant,
            request_queue=self.request_queue
        )

    async def _batch_processor(self):
        """Process policy-value network requests in batches."""
        batch, futures = [], []
        print("Starting batch processor")
        
        while True:
            try:
                # Get next request with timeout
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=self.batch_interval
                )
                batch.append((request[0], request[1]))
                futures.append(request[2])
                
                # Process batch if full or queue is empty
                if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                    # Update API calls count
                    self.total_api_calls += len(batch)
                    
                    # Process current batch
                    current_batch, current_futures = batch, futures
                    batch, futures = [], []  # Reset for next batch
                    asyncio.create_task(self._process_network_requests(current_batch, current_futures))
                
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.001)

    async def _process_network_requests(self, batch: list, futures: list):
        """Process batch of requests through policy-value network."""
        try:
            # Get predictions from policy-value network
            results = self.policy_value_fn(batch)
            
            # Distribute results to waiting futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            print(f"Network request error: {e}")
            self._handle_batch_error(futures, e)

    def _handle_batch_error(self, futures: list, error: Exception):
        """Handle errors in batch processing."""
        for future in futures:
            if not future.done():
                future.set_exception(error)

    async def _run_tree_spot(self, spot_index: int):
        """Manage a single tree spot in the forest."""
        while True:
            current_question = None
            try:
                # Get current tree and process it
                tree = self.trees[spot_index]
                current_question = tree.question
                policy_data, value_data = await tree.search()
                
                # Process collected trajectories
                await self._process_trajectories(current_question, policy_data, value_data)
                
                # Update tree with next question
                next_question = await self._select_next_question(current_question)
                await self._update_tree_spot(spot_index, next_question)
                
            except Exception as e:
                await self._handle_spot_error(spot_index, current_question, e)
                await asyncio.sleep(1)

    async def _process_trajectories(self, question: str, policy_data: list, value_data: list):
        """Process and store trajectories as training data."""

        unique_value_data = list(set(value_data))
        if policy_data:
            policy_data = random.choices(policy_data, k=len(unique_value_data))
        
        processed_policy = self.process_policy_trajectory(question, policy_data)
        processed_value = self.process_value_trajectory(question, unique_value_data)
        # Store processed data
        self.policy_training_data.extend(processed_policy)
        self.value_training_data.extend(processed_value)
        
        # Update counts
        self.policy_data_counts[question] += len(processed_policy)
        self.value_data_counts[question] += len(processed_value)

    async def _select_next_question(self, current_question: str) -> str:
        """Select next question to process based on data counts."""
        async with self.active_questions_lock:
            # Remove current question from active set
            if current_question in self.active_questions:
                self.active_questions.remove(current_question)
            
            # Select question with least data from inactive or all questions
            inactive = [(q, c) for q, c in self.value_data_counts.items() 
                       if q not in self.active_questions]
            candidates = inactive or self.value_data_counts.items()
            next_question = min(candidates, key=lambda x: x[1])[0]
            
            # Mark new question as active
            self.active_questions.add(next_question)
            return next_question

    async def _update_tree_spot(self, spot_index: int, question: str):
        """Update tree spot with new question."""
        question_idx = list(self.policy_data_counts.keys()).index(question)
        self.trees[spot_index] = self._create_tree(question_idx, question)

    async def _handle_spot_error(self, spot_index: int, question: str, error: Exception):
        """Handle errors in tree spot processing."""
        print(f"Spot {spot_index} error: {type(error).__name__}: {str(error)}")
        print(f"Current question: {question}")
        print(f"Stack trace:", exc_info=True)
        
        try:
            async with self.active_questions_lock:
                if question and question in self.active_questions:
                    self.active_questions.remove(question)
        except Exception as lock_error:
            print(f"Error cleaning up active questions: {type(lock_error).__name__}: {str(lock_error)}")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        batch_processor = asyncio.create_task(self._batch_processor())
        tree_spots = [self._run_tree_spot(i) for i in range(len(self.trees))]
        await asyncio.gather(batch_processor, *tree_spots)
    
    

