from typing import Self, Callable, List, Tuple, TypeVar, Optional, Dict
import random
import re
import time
import asyncio

class Node:
    def __init__(self, state: str, parent: Self | None, N: int, Q: float, V: float):
        """
        Initialize a node in the MCTS tree.
        Args:
            state: The state representation as a string
            parent: Parent node in the tree (None for root)
            N: Number of visits to this node
            Q: Action value estimate
            V: Initial value estimate
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.N = N
        self.Q = Q
        self.V = V
        self.is_terminal = self.state.endswith('.') or self.state.endswith('.\n')

    @property
    def is_visited(self) -> bool:
        return self.N > 0

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    def expand(self, children_values: list[tuple[str, float]]):
        """
        Create child nodes for this node.
        Args:
            children_values: List of (state, value) pairs for child nodes
        """
        for child, V in children_values:
            self.children.append(Node(state=child, parent=self, N=0, Q=V, V=V))

    def evaluate_node(self, question: str) -> float:
        """
        Evaluate if this state solves the 24 game.
        Args:
            question: Question string (e.g., "4 4 6 8")
        Returns:
            1.0 if expression evaluates to 24 using correct numbers
            0.0 otherwise
        """
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

class MCTS:
    def __init__(self, V_root: float, question: str, max_expansions: int, max_leaves: int, c_explore: float,
                 process_policy_data: Callable[[str, str], str],
                 process_value_data: Callable[[str, str, float], str],
                 request_queue):
        """
        Initialize MCTS search for use within a forest with batched processing.
        
        Args:
            V_root: Initial value estimate for root node
            question: Question being answered (e.g., "4 4 6 8" for 24 game)
            max_expansions: Maximum number of node expansions to perform
            max_leaves: Maximum number of leaves to expand
            c_explore: Exploration constant for UCB formula
            process_policy_data: Function to process policy data for training
            process_value_data: Function to process value data for training
            request_queue: Queue for batched API requests (required)
        """
        self.root = Node(state="", parent=None, N=0, Q=0, V=V_root)
        self.question = question
        self.max_expansions = max_expansions
        self.max_leaves = max_leaves
        self.c_explore = c_explore
        self.process_policy_data = process_policy_data
        self.process_value_data = process_value_data
        self.policy_training_data = []
        self.value_training_data = []
        self.request_queue = request_queue
        self.expansion_count = 0
       
    async def _get_policy_value(self, current_node) -> list[tuple[str, float]]:
        future = asyncio.Future()
        await self.request_queue.put((self.question, current_node.state, future))
        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            print(f"Error: No response after 60s at {current_node.state}.")
            return []

    async def mcts(self):
        current_node = self.root
        while self.expansion_count < self.max_expansions and self.root.N < self.max_leaves:
            if current_node.has_children:
                current_node = self.step_forward(current_node)
            elif current_node.is_terminal:
                value = current_node.evaluate_node(self.question)
                self.value_training_data.append((current_node.state, value))
                if value:    
                    self.policy_training_data.append(current_node.state)
                self.backpropagate(current_node, value, True)
                current_node = self.root
            elif not current_node.is_visited:
                self.backpropagate(current_node, current_node.V, False)
                current_node = self.root
            else:
                try:
                    children_values = await self._get_policy_value(current_node)
                    current_node.expand(children_values)
                    self.expansion_count += 1  # Count each expansion
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break
            await asyncio.sleep(0)
        print(f"Tree completed with {self.expansion_count} expansions and {self.root.N} leaves.")
        return self.policy_training_data, self.value_training_data
    
    def step_forward(self, node: Node) -> Node:
        """
        Select best child node using UCB1 formula.
        """
        ucb1 = lambda child: child.Q + self.c_explore * (node.N ** 0.5) / (child.N + 1)
        return max(node.children, key=ucb1)

    def backpropagate(self, node: Node, value: float, is_terminal: bool):
        """
        Update node statistics from current node to root.
        
        Args:
            node: Starting node for backpropagation
            value: Value to backpropagate
            is_terminal: Whether the node is a terminal state
        """
        while node:
            node.N += 1
            node.Q += (value - node.Q) / node.N
            node = node.parent

class MCTS_forest:
    def __init__(self, V_initials: list[float], questions: list[str],
                 max_expansions: int, max_leaves: int, num_trees: int, c_explore: float,
                 policy_value_fn: Callable[[List[Tuple[str, str]]], List[List[Tuple[str, float]]]],
                 process_policy_data: Callable[[str, str], str],
                 process_value_data: Callable[[str, str, float], str],
                 batch_size: int = 8,
                 batch_interval: float = 0.1):
        """
        Initialize a forest of MCTS trees.
        Args:
            V_initials: Initial value estimates for root nodes
            questions: Questions being answered
            max_expansions: Maximum number of node expansions to perform per tree
            max_leaves: Maximum number of leaves to expand per tree
            num_trees: Number of trees to run concurrently
            c_explore: Exploration constant for UCB formula
            policy_value_fn: Function that takes a list of (question, state) pairs and returns a list of lists of (next_state, value) pairs
            process_policy_data: Function to process policy data for training
            process_value_data: Function to process value data for training
            batch_size: Maximum number of API requests to batch together
            batch_interval: Time interval (in seconds) for processing batched requests
        """
        self.V_initials = V_initials
        self.questions = questions
        self.max_expansions = max_expansions
        self.max_leaves = max_leaves
        self.num_trees = num_trees
        self.c_explore = c_explore
        self.policy_value_fn = policy_value_fn
        self.process_policy_data = process_policy_data
        self.process_value_data = process_value_data
        self.policy_training_data = []
        self.value_training_data = []
        self.policy_data_counts = {q: 0 for q in questions}
        self.value_data_counts = {q: 0 for q in questions}
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.active_questions = set()
        self.active_questions_lock = asyncio.Lock()
        self.trees = []
        for i in range(num_trees):
            question_idx = i % len(self.questions)
            question = self.questions[question_idx]
            self.active_questions.add(question)
            self.trees.append(
                MCTS(V_root=self.V_initials[question_idx], question=question,
                     max_expansions=self.max_expansions, max_leaves=self.max_leaves, c_explore=self.c_explore,
                     process_policy_data=self.process_policy_data,
                     process_value_data=self.process_value_data,
                     request_queue=self.request_queue)
            )

    async def _batch_processor(self):
        """Process API requests in batches without waiting for completion."""
        batch, futures = [], []
        last_batch_time = time.time()  # Track when the last batch was processed
        print("Batch processor started")
        while True:
            try:
                item = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_interval)
                batch.append((item[0], item[1]))
                futures.append(item[2])
            except asyncio.TimeoutError:
                pass
            if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                current_time = time.time()
                time_between_batches = current_time - last_batch_time
                seconds_per_call = time_between_batches / len(batch) if batch else 0
                print(f"Batch size: {len(batch)}, Time between batches: {time_between_batches:.4f} seconds, {seconds_per_call:.4f} seconds per call")
                last_batch_time = current_time

                current_batch, current_futures = batch, futures
                batch, futures = [], []

                asyncio.create_task(self._process_batch(current_batch, current_futures))
            
            await asyncio.sleep(0.001)

    async def _process_batch(self, batch, futures):
        """Process a batch and distribute results."""
        try:
            results = self.policy_value_fn(batch)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            print(f"Batch error: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            for future in futures:
                if not future.done():
                    future.set_exception(Exception("Batch processing failed to complete"))

    async def run_tree_spot(self, spot_index: int):
        """Run a tree spot indefinitely, replacing trees as they complete."""
        while True:
            try:
                tree = self.trees[spot_index]
                question = tree.question
                policy_data, value_data = await tree.mcts()
                value_data = list(set(value_data))

                # Process the collected data
                processed_policy_data = self.process_policy_data(question, policy_data)
                processed_value_data = self.process_value_data(question, value_data)
                
                self.policy_training_data.extend(processed_policy_data)
                self.value_training_data.extend(processed_value_data)
                self.policy_data_counts[question] += len(processed_policy_data)
                self.value_data_counts[question] += len(processed_value_data)
                
                async with self.active_questions_lock:
                    self.active_questions.remove(question)
                    inactive = [(q, c) for q, c in self.value_data_counts.items() 
                               if q not in self.active_questions]
                    candidates = inactive or self.value_data_counts.items()
                    next_q = min(candidates, key=lambda x: x[1])[0]
                    self.active_questions.add(next_q)
                
                question_idx = list(self.policy_data_counts.keys()).index(next_q)
                self.trees[spot_index] = MCTS(
                    V_root=self.V_initials[question_idx], question=next_q,
                    max_expansions=self.max_expansions, max_leaves=self.max_leaves, c_explore=self.c_explore,
                    process_policy_data=self.process_policy_data,
                    process_value_data=self.process_value_data,
                    request_queue=self.request_queue
                )
            except Exception as e:
                print(f"Spot {spot_index} error: {e}")
                try:
                    async with self.active_questions_lock:
                        self.active_questions.remove(self.trees[spot_index].question)
                except KeyError:
                    pass
                await asyncio.sleep(1)

    async def run_forest(self):
        """Run batch processor and all tree spots concurrently"""
        processor = asyncio.create_task(self._batch_processor())
        spots = [self.run_tree_spot(i) for i in range(len(self.trees))]
        await asyncio.gather(processor, *spots)
    
    

