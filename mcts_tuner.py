from mcts_search import MCTSNode
from config_tuning import get_config
from typing import Callable, List, Dict
import asyncio
import json
import time
from datetime import datetime
from policy_value_fn import PolicyValueModel
import sys
import re
import os

class MCTSTree_Tuner:
    """Single MCTS tree for exploring solutions to a question."""
    
    def __init__(self, root_value: float, question: str, 
                 exploration_constant: float, max_expansions: int,
                 request_queue, tree_id: int):
        self.root = MCTSNode(state="", parent=None, visit_count=0, 
                            action_value=root_value, value_estimate=root_value)
        self.question = question
        self.exploration_constant = exploration_constant
        self.request_queue = request_queue
        self.tree_id = tree_id
        self.expansion_count = 0
        self.non_terminal_leaves = [self.root]
        self.terminal_leaves = []
        self.favourite_trajectories = []
        self.max_expansions = max_expansions

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
       
        success = int(current.evaluate_terminal_state(self.question)) if current.is_terminal else 0
        self.favourite_trajectories.append((self.expansion_count, success))

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while self.non_terminal_leaves and self.expansion_count < self.max_expansions:
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

        if not self.non_terminal_leaves:
            best_terminal = max(self.terminal_leaves, key=lambda node: node.value_estimate)
            success = best_terminal.evaluate_terminal_state(self.question)
            self.favourite_trajectories.append((self.expansion_count, int(success)))
        else:
            self.get_favourite_trajectory()
        
        return self.favourite_trajectories
    


class MCTSForest_Tuner:
    """Parallel MCTS framework for hyperparameter optimization."""
    def __init__(self, initial_values: list[float], questions: list[str],
                 num_trees: int, exploration_constants: list[float], 
                 branch_factors: list[int], temperatures: list[float],
                 max_forward_passes: int, policy_value_fn: Callable, 
                 batch_size: int, batch_interval: float):
        # Core parameters
        self.questions = questions
        self.initial_values = initial_values
        self.num_trees = num_trees
        self.policy_value_fn = policy_value_fn
        self.max_forward_passes = max_forward_passes

        # Hyperparameters to evaluate
        self.exploration_constants = exploration_constants
        self.branch_factors = branch_factors
        self.temperatures = temperatures
        
        # Batch processing setup
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.total_api_calls = 0
        
        # Results tracking - initialize as empty dictionary
        self.results = {}
        self.completed_count = 0 
        
        # Track config for each tree
        self.tree_configs = {}
        
        # Add list of configurations to process
        self.left_configurations = [(q, e, b, t) for q in questions 
                                  for e in exploration_constants 
                                  for b in branch_factors 
                                  for t in temperatures]
        self.config_lock = asyncio.Lock()
        
        # Initialize trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree_Tuner]:
        """Initialize trees for empty configurations."""
        trees = []
        for idx in range(min(self.num_trees, len(self.left_configurations))):
            # Use the configuration without popping it
            config = self.left_configurations[idx]
            tree = self._create_tree(config, idx)
            if tree:
                self.tree_configs[idx] = config
                trees.append(tree)
        return trees

    def _create_tree(self, config: tuple, tree_idx: int) -> MCTSTree_Tuner:
        """Create tree with configuration-specific policy function."""
        question, exploration_constant, branch_factor, _ = config
        question_idx = self.questions.index(question)
        max_expansions = self.max_forward_passes // branch_factor
        return MCTSTree_Tuner(
            root_value=self.initial_values[question_idx],
            question=question,
            exploration_constant=exploration_constant,
            max_expansions=max_expansions,
            request_queue=self.request_queue,
            tree_id=tree_idx
        )

    async def _process_network_requests(self, batch: list, futures: list):
        """Process batch of requests through policy-value network."""
        try:
            results = self.policy_value_fn(batch)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            print(f"Network request error: {e}")
            self._handle_batch_error(futures, e)

    async def _batch_processor(self):
        """Process policy-value network requests in batches."""
        batch, futures = [], []
        last_activity_time = time.time()
        
        # Calculate total configurations
        total_configs = len(self.questions) * len(self.exploration_constants) * \
                       len(self.branch_factors) * len(self.temperatures)
        
        while True:
            # Exit if all configurations have been processed
            if self.completed_count >= total_configs:
                print("Batch processor: All configurations processed, exiting")
                break
            
            try:
                try:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_interval)
                    last_activity_time = time.time()
                    question, state, tree_id, future = request
                    
                    # Check if tree_id is still valid (might have been removed)
                    if tree_id not in self.tree_configs:
                        print(f"Warning: Request for unknown tree_id {tree_id}")
                        future.set_result([])  # Resolve with empty result
                        continue
                        
                    batch.append((question, state, *self.tree_configs[tree_id][2:]))
                    futures.append(future)
                    
                    if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                        self.total_api_calls += len(batch)
                        asyncio.create_task(self._process_network_requests(batch.copy(), futures.copy()))
                        batch, futures = [], []
                except asyncio.TimeoutError:
                    # Process any remaining items in batch
                    if batch:
                        self.total_api_calls += len(batch)
                        asyncio.create_task(self._process_network_requests(batch.copy(), futures.copy()))
                        batch, futures = [], []
                    
                    # Check for long inactivity
                    if time.time() - last_activity_time > 60 and self.completed_count >= 2600:
                        print(f"WARNING: Batch processor inactive for >60s, completed: {self.completed_count}")
                    
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"Batch processor error: {type(e).__name__}: {str(e)}")
                # Resolve any pending futures to prevent hanging
                for future in futures:
                    if not future.done():
                        future.set_result([])
                batch, futures = [], []
                await asyncio.sleep(1)  # Wait a bit before continuing

    async def _run_tree_spot(self, spot_index: int):
        """Run evaluations for a single tree spot."""
        # Calculate total configurations correctly
        total_configs = len(self.questions) * len(self.exploration_constants) * \
                       len(self.branch_factors) * len(self.temperatures)
        
        while True:
            config = None
            try:
                async with self.config_lock:
                    if not self.left_configurations:
                        break
                    config = self.left_configurations.pop(0)
                
                self.tree_configs[spot_index] = config
                self.results[config] = await self._create_tree(config, spot_index).search()
                self.completed_count += 1
                
                # Check if all configurations have been processed
                if self.completed_count >= total_configs:
                    print(f"Spot {spot_index}: All configurations completed, exiting")
                    break
                    
            except Exception as e:
                if config:
                    print(f"Spot {spot_index} error with config {config}: {type(e).__name__}: {str(e)}")
                    # Mark this configuration as completed even though it failed
                else:
                    print(f"Spot {spot_index} error: {type(e).__name__}: {str(e)}")
                await asyncio.sleep(1)

    async def run_forest(self):
        """Run parallel evaluation of all configurations."""
        batch_processor = asyncio.create_task(self._batch_processor())
        tree_spots = [self._run_tree_spot(i) for i in range(len(self.trees))]
        
        # Calculate total configurations for debugging
        total_configs = len(self.questions) * len(self.exploration_constants) * \
                       len(self.branch_factors) * len(self.temperatures)
        
        # Add periodic status check
        status_task = asyncio.create_task(self._periodic_status_check(tree_spots))
        
        try:
            await asyncio.gather(batch_processor, *tree_spots)
        except Exception as e:
            print(f"Error in run_forest: {type(e).__name__}: {str(e)}")
        finally:
            status_task.cancel()
        
        return self.results

    async def _periodic_status_check(self, tree_spots):
        """Periodically check and print status of tree spots."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            done_count = sum(1 for spot in tree_spots if spot.done())
            print(f"Status check: {done_count}/{len(tree_spots)} tree spots done, {self.completed_count} configs completed")
            


    def _handle_batch_error(self, futures: list, error: Exception):
        """Handle errors in batch processing by resolving futures with empty results."""
        print(f"Handling batch error: {error}")
        for future in futures:
            if not future.done():
                # Resolve the future with an empty result to prevent hanging
                future.set_result([])



class Run_MCTS_Tuner:
    """Runs hyperparameter tuning experiments using MCTS."""
    
    def __init__(self, config: Dict):
        """Initialize tuning experiment with configuration."""
        self.config = config
        self.questions = self._load_questions()
        
        # Create a single PolicyValueModel instance
        api_config = self.config['api']
        forest_config = self.config['forest']
        self.model = PolicyValueModel(
            openai_api_base=api_config['openai_api_base'],
            openai_api_key=api_config['openai_api_key'],
            value_api_base_url=api_config['value_api_base_url'],
            policy_model=api_config['policy_model'],
            max_workers_policy=forest_config.get('max_workers_policy'),
            max_workers_value=forest_config.get('max_workers_value')
        )
        
        self.forest = self._initialize_tuner()
        self.is_running = False
        self.tuning_task = None
        self.monitor_task = None
        self.start_time = None

    # ===== Initialization Methods =====
    
    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        questions_path = self.config['paths']['questions_path']
        with open(questions_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _initialize_tuner(self) -> MCTSForest_Tuner:
        """Initialize MCTS forest tuner with initial value estimates."""
        # Get initial value estimates for all questions
        initial_states = [(q, "") for q in self.questions]
        initial_values = self.model.batch_value_estimate(initial_states)
        
        # Configure and return the forest tuner
        forest_config = self.config['forest']
        return MCTSForest_Tuner(
            initial_values=initial_values,
            questions=self.questions,
            num_trees=forest_config['num_trees'],
            exploration_constants=forest_config['exploration_constants'],
            branch_factors=forest_config['branch_factors'],
            temperatures=forest_config['temperatures'],
            max_forward_passes=forest_config['max_forward_passes'],
            policy_value_fn=self.model.get_policy_value,
            batch_size=forest_config['batch_size'],
            batch_interval=forest_config['batch_interval']
        )
    
    # ===== Monitoring and Statistics Methods =====
    
    def _print_tuning_stats(self) -> None:
        """Print current tuning progress."""
        if not self.start_time:
            return
            
        runtime = time.time() - self.start_time
        completed = self.forest.completed_count
        total = len(self.forest.questions) * len(self.forest.exploration_constants) * \
                len(self.forest.branch_factors) * len(self.forest.temperatures)
        
        if runtime > 0:  # Avoid division by zero
            throughput = self.forest.total_api_calls / runtime
        else:
            throughput = 0
        
        print(f"\n--- Tuning progress after {runtime:.1f} seconds ---")
        print(f"Configurations evaluated: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"Total API calls: {self.forest.total_api_calls}")
        print(f"API throughput: {throughput:.1f} calls/sec")

    async def _monitor_tuning(self) -> None:
        """Monitor tuning progress and stop when tuning task completes."""
        stats_interval = self.config['intervals']['stats_interval']
        export_interval = self.config['intervals']['export_interval']
        last_stats = last_export = time.time()
        
        while self.is_running:
            # Check if tuning task has completed
            if self.tuning_task and self.tuning_task.done():
                print("\nTuning task completed. Stopping monitoring...")
                self.is_running = False
                break
            
            current_time = time.time()
            
            if current_time - last_stats >= stats_interval:
                self._print_tuning_stats()
                last_stats = current_time
            
            if current_time - last_export >= export_interval and self.forest.results:
                await self._save_tuning_results()
                last_export = current_time
            
            await asyncio.sleep(1)
    
    # ===== Data Management Methods =====
    async def _save_tuning_results(self) -> None:
        """Save completed hyperparameter tuning results to file and clear from memory."""
        try:
            if not self.forest.results:
                print("No results to save.")
                return
            
            # Convert tuple keys to structured dictionaries for JSON serialization
            serializable_results = []
            for config_tuple, result in self.forest.results.items():
                # Unpack the tuple into named components
                question, exploration_constant, branch_factor, temperature = config_tuple
                
                # Create a structured entry with compact result representation
                entry = {
                    "config": {
                        "question": question,
                        "exploration_constant": exploration_constant,
                        "branch_factor": branch_factor,
                        "temperature": temperature
                    },
                    # Convert result to a more compact representation
                    # Instead of array of [expansion, success] pairs, use a string
                    "result": ";".join(f"{exp},{success}" for exp, success in result)
                }
                serializable_results.append(entry)
            
            # Save to file
            results_path = self.config['paths']['results_path']
            existing_results = []
            
            # Try to read existing results if file exists
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        existing_results = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not read existing results: {e}")
            
            # Combine existing and new results
            all_results = existing_results + serializable_results
            
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Reset results to empty dictionary
            result_count = len(self.forest.results)
            self.forest.results = {}
            
            print(f"Saved {result_count} configurations to {results_path}")
        except Exception as e:
            print(f"Error saving tuning results: {e}")
    
    # ===== Control Methods =====
    
    async def start_tuning(self) -> None:
        """Start the hyperparameter tuning process."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            print("Starting hyperparameter tuning...")
            self.tuning_task = asyncio.create_task(self.forest.run_forest())
            self.monitor_task = asyncio.create_task(self._monitor_tuning())
            
            # Wait for monitoring task to complete
            await self.monitor_task
            
            # Save any remaining results
            await self._save_tuning_results()
            
            # Check if tuning task had any exceptions
            if self.tuning_task and self.tuning_task.done():
                try:
                    # This will re-raise any exception from the tuning task
                    self.tuning_task.result()
                except Exception as e:
                    print(f"\nTuning task failed with error: {e}")
                    raise
            
        except asyncio.CancelledError:
            print("\nTuning was cancelled.")
        except Exception as e:
            print(f"\nError during tuning: {e}")
        finally:
            self.is_running = False

    async def stop_tuning(self) -> None:
        """Stop the hyperparameter tuning process."""
        if not self.is_running:
            return
        
        print("\nStopping tuning...")
        self.is_running = False
        
        for task in [self.monitor_task, self.tuning_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._save_tuning_results()
        print("\nTuning stopped.")


async def main():
    """Main entry point for hyperparameter tuning."""
    config = get_config()
    # Initialize tuner
    tuner = Run_MCTS_Tuner(config=config)
    try:
        await tuner.start_tuning()
    except KeyboardInterrupt:
        print("\nTuning interrupted by user.")
        await tuner.stop_tuning()

if __name__ == "__main__":
    asyncio.run(main())

