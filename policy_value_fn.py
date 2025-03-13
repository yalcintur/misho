import concurrent.futures
from openai import OpenAI
import requests
from typing import List, Tuple, Any
from urllib.parse import urljoin


class PolicyValueModel:
    """Handles policy (state generation) and value estimation for RL-based math problem solving."""
    
    def __init__(
        self, 
        openai_api_base: str,
        openai_api_key: str = "sk-placeholder",
        value_api_base_url: str = None,
        policy_model: str = "lakomey/sft-135-iter1-10-b32", #"mstojkov/sft-135-checkpoint-3000-improved_policy",
        max_workers_policy: int = 80,
        max_workers_value: int = 30
    ):
        """Initialize policy and value networks with API settings."""
        # Policy network (LLM) setup
        self.policy_network = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
        self.policy_model = policy_model
        
        # Value network setup
        self.value_network_url = value_api_base_url
        
        # Thread pool settings
        self.max_workers_policy = max_workers_policy
        self.max_workers_value = max_workers_value

    def sample_policy(self, question: str, state: str, n_samples: int, temperature: float) -> List[str]:
        """Sample next states from policy network (LLM)."""
        try:
            response = self._query_policy_network(question, state, n_samples, temperature)
            if not response:
                return []
            
            # Add newline for non-terminal states
            if len(state.split("\n")) < 4:
                suffix = "\n"
            elif len(state.split("\n")) >= 4 and not state.rstrip().endswith("."):
                suffix = "."
            else:
                suffix = ""
            
            return list(set(
                (state + action.message.content.strip() + suffix)
                for action in response.choices 
                if hasattr(action, 'message')
            ))
            
        except Exception as e:
            print(f"Policy sampling error: {str(e)}")
            return []

    def estimate_value(self, question: str, state: str) -> float:
        """Estimate state value using value network."""
        
        if self.value_network_url is None:
            return 0.5

        try:
            prompt = question + "\n" + state
            payload = {
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }
        
            response = requests.post(
                url=self.value_network_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return float(response.json().get("value", 0.5))
            else:
                return 0.5
                
        except Exception as e:
            print(f"Value estimation error: {type(e).__name__}: {str(e)}")
            return 0.5


    def parallel_process(self, fn, items: List[Tuple], max_workers: int) -> List[Any]:
        """Process items in parallel using thread pool."""
        if not items:
            return []
            
        results = [None] * len(items)
        workers = min(max_workers, len(items))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fn, *item): i for i, item in enumerate(items)}
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    results[idx] = fn.__defaults__[0] if fn.__defaults__ else None
        
        return results

    def batch_value_estimate(self, questions_and_states: List[Tuple[str, str]]) -> List[float]:
        """Batch estimate values for multiple states."""
        return [] if not questions_and_states else self.parallel_process(
            self.estimate_value, questions_and_states, self.max_workers_value
        )

    def get_policy_value(self, questions_states_params: List[Tuple[str, str, int, float]]) -> List[List[Tuple[str, float]]]:
        """Sample actions from policy and estimate their values.
        Args:
            questions_states_params: List of (question, state, branch_factor, temperature) tuples
        """
        if not questions_states_params:
            return []
        
        # Sample actions from policy
        # Fixed order: (question, state, branch_factor, temperature)
        policy_inputs = [(q, s, n, temp) for q, s, n, temp in questions_states_params]
        next_states = self.parallel_process(self.sample_policy, policy_inputs, self.max_workers_policy)
    
        # Prepare value estimation inputs
        value_inputs, positions = [], []
        questions = [q for q, _, _, _ in questions_states_params]
        
        for i, (question, states) in enumerate(zip(questions, next_states)):
            for j, state in enumerate(states):
                value_inputs.append((question, state))
                positions.append((i, j))
        
        # Get value estimates
        values = self.parallel_process(self.estimate_value, value_inputs, self.max_workers_value)
        
        # Combine results
        result = [[] for _ in range(len(questions_states_params))]
        for (i, j), value in zip(positions, values):
            result[i].append((next_states[i][j], value))
        
        return result

    def _query_policy_network(self, question: str, state: str, n_samples: int, temperature: float):
        """Query the policy network (LLM) for next actions."""
        try:
            prompt = f"{question}\n{state}"
            return self.policy_network.chat.completions.create(
                model=self.policy_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=n_samples,
                max_completion_tokens=90,
                stop=["\n"]
            )
        except Exception as e:
            print(f"Policy network error: {e}")
            return None


if __name__ == "__main__":
    model = PolicyValueModel(
        openai_api_base="http://172.81.127.5:31540/v1",
        value_api_base_url = "http://38.29.145.26:40651/predict" #"http://185.113.120.195:50005/predict"
    )
    
    # Test trajectory
    question = "3 7 11 12"
    states = [
        "",  # Initial state
        "11-12=-1 (left: 3, 7, -1)\n",  # Action 1
        "11-12=-1 (left: 3, 7, -1)\n3*7=21 (left: -1, 21)\n",  # Action 2
        "11-12=-1 (left: 3, 7, -1)\n3*7=21 (left: -1, 21)\n21--1=22 (left: 22)\n"  # Action 3
    ]
    
    # Get policy samples and their values
    results = model.get_policy_value([(question, states[1], 5, 0.7)])
    for result_list in results:
        for next_state, value in result_list:
            print(f"{next_state}{value}")