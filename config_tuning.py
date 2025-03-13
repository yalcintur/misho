"""
Configuration for MCTS hyperparameter tuning.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration for tuning
    'forest': {
        'num_trees': 100,  # Number of parallel trees to run
        'exploration_constants': [0.1, 0.3, 0.5],  # Different exploration constants to evaluate
        'branch_factors': [2, 3, 5],  # Different branching factors to evaluate
        'temperatures': [0.5, 0.7, 0.9],  # Different temperatures to evaluate
        'batch_size': 50,  # Batch size for policy-value network requests
        'batch_interval': 1.0,  # Time to wait for batching requests
        'max_workers_policy': 50,  # Max workers for policy network
        'max_workers_value': 50,  # Max workers for value network
    },
    
    # File paths configuration
    'paths': {
        'questions_path': 'data_val.txt',  # Questions to use for tuning
        'results_path': 'tuning_results.json',  # Where to save tuning results
    },
    
    # Intervals configuration
    'intervals': {
        'export_interval': 60,  # 1 minute - save results periodically
        'stats_interval': 30     # 30 seconds - print stats periodically
    },
    
    # API configuration
    'api': {
        'openai_api_base': "http://172.81.127.5:31540/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': None,  # "http://47.186.25.253:53620/predict"
        'policy_model': "lakomey/sft-135-iter1-10-b32"
    }
}

def get_config():
    """
    Get configuration dictionary for hyperparameter tuning.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy()
