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
        for state, value in state_values:
            self.children.append(MCTSNode(
                state=state, 
                parent=self, 
                visit_count=0, 
                action_value=value, 
                value_estimate=value
            ))

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
            if expr_nums != question_nums or abs(eval(parts[0]) - 24) > 1e-6:
                return 0.0
            return 1.0
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
    
    def __init__(self, root_value: float, question: str, exploration_constant: float, request_queue, tree_id: int):
        self.root = MCTSNode(state="", parent=None, visit_count=0, 
                            action_value=0, value_estimate=root_value)
        self.question = question
        self.exploration_constant = exploration_constant
        self.request_queue = request_queue
        self.tree_id = tree_id
        self.expansion_count = 0
        self.non_terminal_leaves = [self.root]
        self.terminal_leaves = []
        self.favourite_trajectories = []

    async def get_action_values(self, node: MCTSNode) -> list[tuple[str, float]]:
        """Get action-value pairs from policy-value network."""
        future = asyncio.Future()
        await self.request_queue.put((self.question, node.state, self.tree_id, future))
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

    def get_favourite_trajectory(self) -> None:
        """Track the favourite trajectory through the tree and record result."""
        current = self.root
        while current.has_children:
            current = current.favourite_child
       
        if current.is_terminal:
            success = current.evaluate_terminal_state(self.question)
            self.favourite_trajectories.append((self.expansion_count, int(success)))
        else:
            self.favourite_trajectories.append((self.expansion_count, 0))

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while self.non_terminal_leaves:
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            else:
                self.get_favourite_trajectory()
                try:
                    new_states = await self.get_action_values(current)
                    current.add_children(new_states)
                    for child in current.children:
                        if child.is_terminal:
                            self.terminal_leaves.append(child)
                        else:
                            self.non_terminal_leaves.append(child)
                    self.non_terminal_leaves.remove(current)
                    self.expansion_count += 1
                except Exception as e:
                    print(f"Expansion error: {e}")
                    break
            await asyncio.sleep(0)
        
        best_terminal = max(self.terminal_leaves, key=lambda node: node.value_estimate)
        success = best_terminal.evaluate_terminal_state(self.question)
        self.favourite_trajectories.append((self.expansion_count, int(success)))
        
        return self.favourite_trajectories
    


class MCTSForest:
    """Forest of MCTS trees for parallel evaluation of hyperparameter configurations."""    
    def __init__(self, initial_values: list[float], questions: list[str],
                 num_trees: int, exploration_constants: list[float], 
                 branch_factors: list[int], temperatures: list[float],
                 policy_value_fn: Callable, batch_size: int, batch_interval: float):
        # Core parameters
        self.questions = questions
        self.initial_values = initial_values
        self.num_trees = num_trees
        self.policy_value_fn = policy_value_fn
        
        # Hyperparameters to evaluate
        self.exploration_constants = exploration_constants
        self.branch_factors = branch_factors
        self.temperatures = temperatures
        
        # Batch processing setup
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.total_api_calls = 0
        
        # Results tracking
        self.results = {(q, e, b, t): [] for q in questions 
                       for e in exploration_constants 
                       for b in branch_factors 
                       for t in temperatures}
        
        # Track config for each tree
        self.tree_configs = {}
        
        # Initialize trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree]:
        """Initialize trees for empty configurations."""
        empty_configs = [(q, e, b, t) for (q, e, b, t), results 
                        in self.results.items() if not results]
        trees = []
        for idx, config in enumerate(empty_configs[:self.num_trees]):
            tree = self._create_tree(config, idx)
            if tree:
                self.tree_configs[idx] = config
                trees.append(tree)
        return trees

    def _create_tree(self, config: tuple, tree_idx: int) -> MCTSTree:
        """Create tree with configuration-specific policy function."""
        question, exp_const, _, _ = config
        question_idx = self.questions.index(question)
        return MCTSTree(
            root_value=self.initial_values[question_idx],
            question=question,
            exploration_constant=exp_const,
            request_queue=self.request_queue,
            tree_id=tree_idx
        )

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

    async def _batch_processor(self):
        """Process policy-value network requests in batches."""
        batch, futures = [], []
        while True:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(), timeout=self.batch_interval)
                # Unpack the request tuple: (question, state, tree_id, future)
                question, state, tree_id, future = request
                # Create policy batch format directly: (question, state, branch_factor, temperature)
                batch.append((question, state, *self.tree_configs[tree_id][2:]))
                futures.append(future)
                
                if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                    self.total_api_calls += len(batch)
                    current_batch, current_futures = batch, futures
                    batch, futures = [], []
                    asyncio.create_task(self._process_network_requests(
                        current_batch, current_futures))
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.001)

    async def _run_tree_spot(self, spot_index: int):
        """Run evaluations for a single tree spot."""
        while True:
            try:
                tree = self.trees[spot_index]
                if not tree:  # No more configurations to evaluate
                    break
                
                # Find current configuration and process it
                current_config = next((config for config, results in self.results.items() 
                                    if not results and config[0] == tree.question), None)
                if current_config:
                    self.results[current_config] = await tree.search()
                
                # Get next empty configuration
                empty_configs = [(q, e, b, t) for (q, e, b, t), results 
                               in self.results.items() if not results]
                if not empty_configs:
                    break
                
                self.trees[spot_index] = self._create_tree(empty_configs[0], None)
                
            except Exception as e:
                print(f"Spot {spot_index} error: {type(e).__name__}: {str(e)}")
                await asyncio.sleep(1)

    async def run_forest(self):
        """Run parallel evaluation of all configurations."""
        batch_processor = asyncio.create_task(self._batch_processor())
        tree_spots = [self._run_tree_spot(i) for i in range(len(self.trees))]
        await asyncio.gather(batch_processor, *tree_spots)
        return self.results