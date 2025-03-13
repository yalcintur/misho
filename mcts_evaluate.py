import asyncio
import traceback
from mcts_search import MCTSNode
from typing import Callable, List, Dict, Tuple
import random
import time
from policy_value_fn import PolicyValueModel
from config_evaluate import get_config

class MCTSTree_Evaluate:
    """Single MCTS tree for exploring solutions to a question."""
    
    def __init__(self, root_value: float, question: str, max_expansions: int, 
                 exploration_constant: float, request_queue):
        self.root = MCTSNode(state="", parent=None, visit_count=0, 
                            action_value=root_value, value_estimate=root_value)
        self.question = question
        self.exploration_constant = exploration_constant
        self.request_queue = request_queue
        self.expansion_count = 0
        self.max_expansions = max_expansions  
        self.non_terminal_leaves = [self.root]
        self.terminal_leaves = []

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

    def get_favourite_trajectory(self) -> int:
        """Track the favourite trajectory through the tree and record result."""
        current = self.root
        while current.has_children:
            current = current.favourite_child
       
        success = int(current.evaluate_terminal_state(self.question)) if current.is_terminal else 0
        return int(success)

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while (self.expansion_count < self.max_expansions and self.non_terminal_leaves):
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            else:
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

        if not self.non_terminal_leaves:
            best_terminal = max(self.terminal_leaves, key=lambda node: node.value_estimate)
            success = best_terminal.evaluate_terminal_state(self.question)
            return int(success)
        else:
            return self.get_favourite_trajectory()
        
class MCTSForest_Evaluate:
    """Forest of MCTS trees for parallel exploration of multiple questions."""
    
    def __init__(self, initial_values: list[float], questions: list[str],
                 max_expansions: int, num_trees: int, 
                 exploration_constant: float, policy_value_fn: Callable,
                 batch_size: int, batch_interval: float):
        # Initialize forest parameters
        self.initial_values = initial_values
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.exploration_constant = exploration_constant
        
        # Set network functions
        self.policy_value_fn = policy_value_fn
        
        # Set up batch processing
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.total_api_calls = 0
        
        # Initialize tracking structures
        self.results = {}
        self.completed_count = 0 
        self.start_time = None
        
        # Add list of configurations to process
        self.left_questions = [q for q in questions]
        self.config_lock = asyncio.Lock()
        
        # Initialize trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree_Evaluate]:
        """Initialize the forest with trees for each spot."""
        trees = []
        for i in range(self.num_trees):
            question_idx = i % len(self.questions)
            question = self.questions[question_idx]
            trees.append(self._create_tree(question_idx, question))
        return trees

    def _create_tree(self, question_idx: int, question: str) -> MCTSTree_Evaluate:
        """Create a new MCTS tree."""
        return MCTSTree_Evaluate(
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
            # Check if we should exit before waiting for a request
            if self.request_queue.empty() and all(task.done() for task in self.tree_spot_tasks):
                break
            
            try:
                # Use a shorter timeout to check exit conditions more frequently
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
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
                # Just continue the loop, which will check exit conditions again
                pass
            
            await asyncio.sleep(0.001)
        
        print("Batch processor shutting down")

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
                success = await tree.search()
                
                # Record result
                self.results[current_question] = success
                self.completed_count += 1
                
                # Update tree with next question or break if no more questions
                next_question = await self._select_next_question()
                if next_question is None:
                    # No more questions to process
                    break
                    
                await self._update_tree_spot(spot_index, next_question)
                
            except Exception as e:
                await self._handle_spot_error(spot_index, current_question, e)
                await asyncio.sleep(1)

    async def _select_next_question(self) -> str:
        """Select next question to process based on available questions."""
        async with self.config_lock:
            # If there are questions left, take one
            if self.left_questions:
                next_question = self.left_questions.pop(0)
                return next_question
            else:
                # No more questions to process
                return None

    async def _update_tree_spot(self, spot_index: int, question: str):
        """Update tree spot with new question."""
        question_idx = self.questions.index(question)
        self.trees[spot_index] = self._create_tree(question_idx, question)

    async def _handle_spot_error(self, spot_index: int, question: str, error: Exception):
        """Handle errors in tree spot processing."""
        print(f"Spot {spot_index} error: {type(error).__name__}: {str(error)}")
        print(f"Current question: {question}")
        traceback.print_exc()

    def get_average_success_rate(self) -> float:
        """Calculate the average success rate across all evaluated questions."""
        if not self.results:
            return 0.0
        
        total_success = sum(self.results.values())
        return total_success / len(self.results)

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        self.start_time = time.time()
        
        # Create and store tasks so we can check their status
        self.tree_spot_tasks = [asyncio.create_task(self._run_tree_spot(i)) for i in range(len(self.trees))]
        batch_processor = asyncio.create_task(self._batch_processor())
        
        # Wait for all tree spots to complete
        await asyncio.gather(*self.tree_spot_tasks)
        
        # Wait for batch processor to finish processing any remaining requests
        await batch_processor
        
        # Return the average success rate when all questions are processed
        return self.get_average_success_rate()


class Run_MCTS_Evaluate:
    def __init__(self, config: Dict):
        self.config = config
        self.questions = self._load_questions()
        self.forest = self._initialize_search()
        self.is_running = False
        self.evaluation_task = None
        self.monitor_task = None

    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        questions_path = self.config['paths']['questions_path']
        with open(questions_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
        
    def _initialize_model(self) -> PolicyValueModel:
        """Initialize policy-value network model."""
        api_config = self.config['api']
        forest_config = self.config['forest']
        
        return PolicyValueModel(
            openai_api_base=api_config['openai_api_base'],
            openai_api_key=api_config['openai_api_key'],
            value_api_base_url=api_config['value_api_base_url'],
            policy_model=api_config['policy_model'],
            max_workers_policy=forest_config['max_workers_policy'],
            max_workers_value=forest_config['max_workers_value']
        )
    
    def _get_initial_values(self, model: PolicyValueModel) -> List[float]:
        """Get initial value estimates for all questions."""
        initial_states = [(q, "") for q in self.questions]
        return model.batch_value_estimate(initial_states)

    def _initialize_search(self) -> MCTSForest_Evaluate:
        """Initialize MCTS forest."""
        policy_value_model = self._initialize_model()
        initial_values = self._get_initial_values(policy_value_model)
        forest_config = self.config['forest']
        
        # Create wrapper function that adds branch_factor and temperature from config
        def policy_value_fn(questions_states: List[Tuple[str, str]]) -> List[List[Tuple[str, float]]]:
            return policy_value_model.get_policy_value([
                (q, s, forest_config['branch_factor'], forest_config['temperature'])
                for q, s in questions_states
            ])
        
        return MCTSForest_Evaluate(
            initial_values=initial_values,
            questions=self.questions,
            max_expansions=forest_config['max_expansions'],
            num_trees=forest_config['num_trees'],
            exploration_constant=forest_config['c_explore'],
            policy_value_fn=policy_value_fn,
            batch_size=forest_config['batch_size'],
            batch_interval=forest_config['batch_interval']
        )
    
    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        runtime = time.time() - self.forest.start_time
        
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"Total API calls: {self.forest.total_api_calls}")
        print(f"API throughput: {self.forest.total_api_calls / runtime if runtime > 0 else 0:.1f} calls/sec")
        print(f"Questions processed: {self.forest.completed_count}/{len(self.questions)}")

    def _check_evaluation_complete(self) -> bool:
        """Check if evaluation targets have been met."""
        return self.forest.completed_count >= len(self.questions)

    async def _monitor_collection(self) -> None:
        """Monitor collection progress and handle periodic tasks."""
        intervals = self.config['intervals']
        last_stats = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_stats >= intervals['stats_interval']:
                self._print_collection_stats()
                last_stats = current_time
            
            if self._check_evaluation_complete():
                print("\nEvaluation complete! All questions have been processed.")
                self.is_running = False
                break
            
            await asyncio.sleep(1)

    async def start_evaluation(self) -> float:
        """Start the evaluation process and return the average success rate."""
        if self.is_running:
            return 0.0
        
        self.is_running = True
        average_success_rate = 0.0
        
        try:
            self.evaluation_task = asyncio.create_task(self.forest.run_forest())
            self.monitor_task = asyncio.create_task(self._monitor_collection())
            
            # Wait for the monitor task to complete
            await self.monitor_task
            
            # Get the result from the evaluation task
            average_success_rate = await self.evaluation_task
            
            print(f"\nEvaluation complete! Average success rate: {average_success_rate:.2%}")
            
        except asyncio.CancelledError:
            print("\nEvaluation was cancelled.")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
        finally:
            self.is_running = False
            
        return average_success_rate

    async def stop_evaluation(self) -> None:
        """Stop the evaluation process."""
        if not self.is_running:
            return
        
        print("\nStopping evaluation...")
        self.is_running = False

        for task in [self.monitor_task, self.evaluation_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


async def main():
    """Main entry point for evaluation."""
    config = get_config()
    
    # Evaluation
    evaluator = Run_MCTS_Evaluate(config=config)
    try:
        print("Starting evaluation...")
        average_success_rate = await evaluator.start_evaluation()
        print(f"Final average success rate: {average_success_rate:.2%}")
    except KeyboardInterrupt:
        await evaluator.stop_evaluation()

if __name__ == "__main__":
    asyncio.run(main())