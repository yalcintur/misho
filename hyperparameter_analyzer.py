import json
from typing import Dict, List, Tuple, Any

def load_hyperparameter_data(results_path: str) -> Dict[Tuple[str, float, int], List[Tuple[int, int]]]:
    """
    Load hyperparameter tuning results into a dictionary.
    
    Args:
        results_path: Path to the JSON results file
        
    Returns:
        Dictionary with keys (question, explore_constant, branch_factor) and 
        values as list of (expansion, success) tuples
    """
    # Load the raw data
    with open(results_path, 'r') as f:
        raw_data = json.load(f)
    
    # Create the dictionary in the requested format
    results_dict = {}
    
    for entry in raw_data:
        config = entry['config']
        
        # Extract key components
        question = config['question']
        explore_constant = config['exploration_constant']
        branch_factor = config['branch_factor']
        temperature = config['temperature']
        
        # Parse trajectory string into list of tuples
        trajectory = []
        if entry['result']:
            for pair in entry['result'].split(';'):
                if ',' in pair:
                    exp, success = pair.split(',')
                    trajectory.append((int(exp), int(success)))
        
        # Store in dictionary with tuple key
        key = (question, explore_constant, branch_factor, temperature)
        results_dict[key] = trajectory
    
    return results_dict

def best_configuration(results_dict, num_forward_passes):
    """
    Find the best hyperparameter configuration that maximizes success across questions
    within the constraint of maximum forward passes.
    
    Args:
        results_dict: Dictionary with keys (question, explore_constant, branch_factor, temperature)
                     and values as list of (expansion, success) tuples
        num_forward_passes: Maximum number of forward passes allowed
        
    Returns:
        Tuple of (exploration_constant, branch_factor, temperature) that maximizes success
    """
    # Group results by configuration (excluding question)
    config_scores = {}
    
    # For each configuration
    for key, trajectory in results_dict.items():
        question, exploration_constant, branch_factor, temperature = key
        config = (exploration_constant, branch_factor, temperature)
        
        # Find the maximum expansion index that satisfies the constraint
        # i * branch_factor <= num_forward_passes
        max_allowed_expansion = num_forward_passes // branch_factor
        
        # Find the success value at the highest valid expansion index
        success_value = 0
        for exp, success in trajectory:
            if exp <= max_allowed_expansion:
                success_value = success  # This will keep the latest valid success
        
        # Add this success value to the configuration's total
        if config not in config_scores:
            config_scores[config] = {}
        config_scores[config][question] = success_value
    
    # Calculate total success for each configuration across all questions
    config_totals = {}
    for config, question_scores in config_scores.items():
        # Normalize by the number of questions if non-zero
        num_questions = len(question_scores)
        if num_questions > 0:
            config_totals[config] = sum(question_scores.values()) / num_questions
        else:
            config_totals[config] = 0
    
    # Find the configuration with the highest total success
    if not config_totals:
        return None  # No valid configurations found
    
    best_config = max(config_totals.items(), key=lambda x: x[1])[0]
    best_score = config_totals[best_config]
    return best_config, best_score

if __name__ == "__main__":
    # Example usage
    results_path = "tuning_results.json"  # Update with your actual file path
    data = load_hyperparameter_data(results_path)
    
    # Print first few entries to verify format
    for i, (key, value) in enumerate(data.items()):
        print(f"Key: {key}")
        print(f"Value: {value}")
        print()
        if i >= 2:  # Just show first 3 entries
            break
    
    # Get the best configuration
    for i in range(7):
        best_config, best_score = best_configuration(data, 2**(i+3))
        print(f"Best configuration for {2**(i+3)} forward passes: {best_config} with score: {best_score}")

    
