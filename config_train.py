"""
Configuration for MCTS training.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration
    'forest': {
        'num_trees': 100,
        'max_expansions': 20,
        'c_explore': 0.3,
        'temperature': 0.7,
        'branch_factor': 3,
        'batch_size': 50,
        'batch_interval': 1.0,
        'max_workers_policy': 50,
        'max_workers_value': 50,
    },
    
    # Training configuration
    'training': {
        'target_examples_per_question': 40
    },
    
    # File paths configuration
    'paths': {
        'questions_path': 'data_train.txt',
        'policy_data_path': 'policy_training_data.jsonl',
        'value_data_path': 'value_training_data.jsonl',
        'stats_path': 'training_stats.json'
    },
    
    # Intervals configuration
    'intervals': {
        'export_interval': 120,  # 2 minutes
        'stats_interval': 60    # 1 minute
    },
    
    # API configuration
    'api': {
        'openai_api_base': "http://109.198.107.223:51181/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': "http://47.186.25.253:53620/predict",  #
        'value_api_endpoint': None
    }
}

def get_config():
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy() 