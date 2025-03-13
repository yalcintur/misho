import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List

from mcts_search import MCTSForest
from policy_value_fn import PolicyValueModel
from trajectory_processor import TrajectoryProcessor
from config_train import get_config


class MCTSDataCollector:
    """Collects and processes MCTS trajectories into training data for policy and value networks."""
    
    def __init__(self, config: Dict):
        """Initialize collector with configuration."""
        self.config = config
        self.questions = self._load_questions()
        self.trajectory_processor = TrajectoryProcessor()
        self.forest = self._initialize_search()
        self.is_running = False
        self.search_task = self.monitor_task = None

    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        with open(self.config['paths']['questions_path'], 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _initialize_model(self) -> PolicyValueModel:
        """Initialize policy-value network model."""
        forest_config = self.config['forest']
        api_config = self.config['api']
        
        return PolicyValueModel(
            openai_api_base=api_config['openai_api_base'],
            openai_api_key=api_config['openai_api_key'],
            value_api_base_url=api_config['value_api_base_url'],
            value_api_endpoint=api_config['value_api_endpoint'],
            temperature=forest_config['temperature'],
            branch_factor=forest_config['branch_factor'],
            max_workers_policy=forest_config['max_workers_policy'],
            max_workers_value=forest_config['max_workers_value']
        )

    def _get_initial_values(self, model: PolicyValueModel) -> List[float]:
        """Get initial value estimates for all questions."""
        initial_states = [(q, "") for q in self.questions]
        return model.batch_value_estimate(initial_states)

    def _initialize_search(self) -> MCTSForest:
        """Initialize MCTS forest."""
        policy_value_model = self._initialize_model()
        initial_values = self._get_initial_values(policy_value_model)
        forest_config = self.config['forest']
        
        return MCTSForest(
            initial_values=initial_values,
            questions=self.questions,
            max_expansions=forest_config['max_expansions'],
            num_trees=forest_config['num_trees'],
            exploration_constant=forest_config['c_explore'],
            policy_value_fn=policy_value_model.get_policy_value,
            process_policy_trajectory=self.trajectory_processor.process_policy_trajectory,
            process_value_trajectory=self.trajectory_processor.process_value_trajectory,
            batch_size=forest_config['batch_size'],
            batch_interval=forest_config['batch_interval']
        )

    async def _save_training_data(self, path: str, data: list, data_type: str) -> None:
        """Save processed training data to file."""
        try:
            with open(path, 'a') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            print(f"{data_type} data exported to {path}")
        except Exception as e:
            print(f"Error exporting {data_type} data: {e}")

    async def _export_training_data(self) -> None:
        """Export processed policy and value training data."""
        await self._save_training_data(
            self.config['paths']['policy_data_path'],
            self.forest.policy_training_data,
            "Policy"
        )
        self.forest.policy_training_data = []
        
        await self._save_training_data(
            self.config['paths']['value_data_path'],
            self.forest.value_training_data,
            "Value"
        )
        self.forest.value_training_data = []

    async def _save_collection_stats(self) -> None:
        """Save current data collection statistics."""
        try:
            runtime = time.time() - self.forest.start_time
            stats = {
                'timestamp': datetime.now().isoformat(),
                'value_data_counts': self.forest.value_data_counts,
                'policy_data_counts': self.forest.policy_data_counts,
                'total_api_calls': self.forest.total_api_calls,
                'runtime': runtime,
                'api_throughput': self.forest.total_api_calls / runtime if runtime > 0 else 0
            }
            with open(self.config['paths']['stats_path'], 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        target = self.config['training']['target_examples_per_question']
        counts = self.forest.value_data_counts.values()
        runtime = time.time() - self.forest.start_time
        
        sufficient = sum(1 for c in counts if c >= target)
        non_zero_value = sum(1 for c in counts if c > 0)
        non_zero_policy = sum(1 for c in self.forest.policy_data_counts.values() if c > 0)
        
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"Total API calls: {self.forest.total_api_calls}")
        print(f"API throughput: {self.forest.total_api_calls / runtime if runtime > 0 else 0:.1f} calls/sec")
        print(f"Questions with target examples ({target}+): {sufficient}/{len(self.questions)} ")
        print(f"Questions with any value/policy examples: {non_zero_value}/{non_zero_policy}")

    def _check_collection_complete(self) -> bool:
        """Check if training data collection targets have been met."""
        target = self.config['training']['target_examples_per_question']
        return all(count >= target for count in self.forest.value_data_counts.values())

    async def _monitor_collection(self) -> None:
        """Monitor collection progress and handle periodic tasks."""
        intervals = self.config['intervals']
        last_export = last_stats = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_stats >= intervals['stats_interval']:
                self._print_collection_stats()
                await self._save_collection_stats()
                last_stats = current_time
            
            if current_time - last_export >= intervals['export_interval']:
                await self._export_training_data()
                last_export = current_time
            
            if self._check_collection_complete():
                print("\nCollection complete! All questions have reached the target.")
                self.is_running = False
                break
            
            await asyncio.sleep(1)

    async def start_collection(self) -> None:
        """Start the data collection process."""
        if self.is_running:
            return
        
        self.is_running = True
        try:
            self.search_task = asyncio.create_task(self.forest.run_forest())
            self.monitor_task = asyncio.create_task(self._monitor_collection())
            await self.monitor_task
            await self._export_training_data()
            await self._save_collection_stats()
        except asyncio.CancelledError:
            print("\nCollection was cancelled.")
        except Exception as e:
            print(f"\nError during collection: {e}")
        finally:
            self.is_running = False

    async def stop_collection(self) -> None:
        """Stop the data collection process."""
        if not self.is_running:
            return
        
        print("\nStopping collection...")
        self.is_running = False
        
        for task in [self.monitor_task, self.search_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._export_training_data()
        await self._save_collection_stats()


async def main():
    """Main entry point for collection."""
    config = get_config()
    collector = MCTSDataCollector(config=config)
    
    try:
        await collector.start_collection()
    except KeyboardInterrupt:
        await collector.stop_collection()


if __name__ == "__main__":
    asyncio.run(main())
