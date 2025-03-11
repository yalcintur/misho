import json
from typing import List, Tuple

class TrajectoryProcessor:
    """Processes trajectories into training data for policy and value networks."""
    
    @staticmethod
    def _create_prompt(question: str, state: List[str] = None) -> List[dict]:
        """Create prompt with question and current state (sequence of actions)."""
        return [{"role": "user", "content": f"{question.rstrip()}\n{(''.join(a.strip() + '\n' for a in state) if state else '')}"}]
    
    @staticmethod
    def _format_data(prompt: List[dict], completion: str) -> str:
        """Format as JSON string for model training."""
        return json.dumps({"prompt": prompt, "completion": [{"role": "assistant", "content": completion}]})

    def process_policy_trajectory(self, question: str, trajectories: List[str]) -> List[str]:
        """Process trajectories for policy network training."""
        all_prompts = []
        for trajectory in trajectories:
            actions = trajectory.strip().split('\n')
            state = []
            for action in actions:
                all_prompts.append(self._format_data(self._create_prompt(question, state), action + "\n"))
                state.append(action)  # state grows as actions are added
        return all_prompts

    def process_value_trajectory(self, question: str, trajectory_reward_pairs: List[Tuple[str, float]]) -> List[str]:
        """Process trajectories for value network training."""
        all_prompts = []
        for trajectory, reward in trajectory_reward_pairs:
            actions = trajectory.strip().split('\n')
            state = []
            for action in actions:
                all_prompts.append(self._format_data(self._create_prompt(question, state), str(reward)))
                state.append(action)
            all_prompts.append(self._format_data(self._create_prompt(question, state), str(reward)))
        return all_prompts


if __name__ == "__main__":
    processor = TrajectoryProcessor()
    question = "3 7 11 12"
    trajectory = "3+11=14 (left: 7, 12, 14)\n7/14=0.5 (left: 12, 0.5)\n12/0.5=24.0 (left: 24.0)\n12/(7/(3+11))=24.0."
    
    print("\nPolicy:", *processor.process_policy_trajectory(question, [trajectory]), sep="\n")
    print("\nValue:", *processor.process_value_trajectory(question, [(trajectory, 1.0)]), sep="\n")

