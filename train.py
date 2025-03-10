import asyncio
import json
import os
import pickle
import time
from datetime import datetime
from functools import partial
from typing import Dict, List

from mcts import MCTS_forest
from policy_value_fn import policy_value_fn
from process_data import process_value_data, process_policy_data
from config import get_config

class MCTSTrainer:
    def __init__(self, questions: List[str], config: Dict):
        self.questions = questions
        self.config = config
        self.forest = self._initialize_forest()
        self.is_running = False
        self.forest_task = self.monitor_task = None
    
    def _initialize_forest(self) -> MCTS_forest:
        checkpoint_path = self.config['paths']['checkpoint_path']
        
        if os.path.exists(checkpoint_path):
            try:
                print(f"Loading existing forest checkpoint from {checkpoint_path}...")
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading forest checkpoint: {e}")
                print("Creating a new forest instead...")
        else:
            print("No existing forest checkpoint found. Creating a new forest...")
        
        forest_config = self.config['forest']
        configured_policy_value_fn = partial(
            policy_value_fn,
            temperature=forest_config['temperature'],
            branch_factor=forest_config['branch_factor']
        )
        
        return MCTS_forest(
            V_initials=[0.5 for _ in range(len(self.questions))],
            questions=self.questions,
            policy_value_fn=configured_policy_value_fn,
            process_policy_data=process_policy_data,
            process_value_data=process_value_data,
            max_expansions=forest_config['max_expansions'],
            max_leaves=forest_config['max_leaves'],
            num_trees=forest_config['num_trees'],
            c_explore=forest_config['c_explore'],
            batch_size=forest_config['batch_size'],
            batch_interval=forest_config['batch_interval']
        )
    
    async def _export_training_data(self):
        policy_path = self.config['paths']['policy_data_path']
        value_path = self.config['paths']['value_data_path']
        
        try:
            with open(policy_path, 'a') as f:
                for data in self.forest.policy_training_data:
                    f.write(json.dumps(data) + '\n')
            self.forest.policy_training_data = []
            print(f"Policy training data exported to {policy_path}")
        except Exception as e:
            print(f"Error exporting policy data: {e}")
        
        try:
            with open(value_path, 'a') as f:
                for data in self.forest.value_training_data:
                    f.write(json.dumps(data) + '\n')
            self.forest.value_training_data = []
            print(f"Value training data exported to {value_path}")
        except Exception as e:
            print(f"Error exporting value data: {e}")
    
    async def _save_checkpoint(self):
        checkpoint_path = self.config['paths']['checkpoint_path']
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.forest, f)
            print(f"Forest checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving forest checkpoint: {e}")
    
    async def _save_stats(self):
        stats_path = self.config['paths']['stats_path']
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'value_data_counts': self.forest.value_data_counts,
                'policy_data_counts': self.forest.policy_data_counts,
                'active_questions': list(self.forest.active_questions)
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Training stats saved to {stats_path}")
        except Exception as e:
            print(f"Error saving training stats: {e}")
    
    def _print_stats(self):
        target = self.config['training']['target_examples_per_question']
        
        sufficient_count = sum(1 for count in self.forest.value_data_counts.values() if count >= target)
        non_zero_value = sum(1 for count in self.forest.value_data_counts.values() if count > 0)
        non_zero_policy = sum(1 for count in self.forest.policy_data_counts.values() if count > 0)
        avg_examples = sum(self.forest.value_data_counts.values()) / len(self.questions)
        
        print(f"\n--- Stats at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Questions with target value examples ({target}+): "
              f"{sufficient_count}/{len(self.questions)} "
              f"({sufficient_count/len(self.questions)*100:.1f}%)")
        print(f"Questions with any value examples: {non_zero_value}/{len(self.questions)}")
        print(f"Questions with any policy examples: {non_zero_policy}/{len(self.questions)}")
        print(f"Average value examples per question: {avg_examples:.1f}")
        print(f"Currently active questions: {len(self.forest.active_questions)}")
    
    def _check_completion(self) -> bool:
        target = self.config['training']['target_examples_per_question']
        return all(count >= target for count in self.forest.value_data_counts.values())
    
    async def _monitor_training(self):
        intervals = self.config['intervals']
        last_save = last_export = last_stats = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_stats >= intervals['stats_interval']:
                self._print_stats()
                last_stats = current_time
            
            if current_time - last_save >= intervals['save_interval']:
                print(f"\nSaving checkpoint at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
                await self._save_checkpoint()
                await self._save_stats()
                last_save = current_time
            
            if current_time - last_export >= intervals['export_interval']:
                print(f"\nExporting data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
                await self._export_training_data()
                last_export = current_time
            
            if self._check_completion():
                print("\nTraining complete! All questions have reached the target.")
                self.is_running = False
                break
            
            await asyncio.sleep(1)
    
    async def start(self):
        if self.is_running:
            print("Training is already running.")
            return
        
        self.is_running = True
        
        try:
            self.forest_task = asyncio.create_task(self.forest.run_forest())
            self.monitor_task = asyncio.create_task(self._monitor_training())
            await self.monitor_task
            await self._save_checkpoint()
            await self._export_training_data()
            await self._save_stats()
            print("Training completed successfully!")
        
        except asyncio.CancelledError:
            print("Training was cancelled.")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            self.is_running = False
    
    async def stop(self):
        if not self.is_running:
            print("Training is not running.")
            return
        
        print("\nStopping MCTS training...")
        self.is_running = False
        
        for task in [self.monitor_task, self.forest_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._save_checkpoint()
        await self._export_training_data()
        await self._save_stats()
        print("Training stopped and data saved.")

async def main():
    # Read questions from text file instead of pickle
    with open('all_questions.txt', 'r') as f:
        questions = [line.strip() for line in f.readlines()]
    
    config = get_config()
    
    print("\n=== MCTS Training Configuration ===")
    for category, params in config.items():
        print(f"\n{category.capitalize()} parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print(f"\nTotal questions: {len(questions)}")
    print("==================================\n")
    
    trainer = MCTSTrainer(questions=questions, config=config)
    
    try:
        await trainer.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected.")
        await trainer.stop()

if __name__ == "__main__":
    asyncio.run(main())
