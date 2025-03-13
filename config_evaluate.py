"""
Configuration for MCTS training.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration
    'forest': {
        'num_trees': 100,
        'max_expansions': 64,
        'c_explore': 0.3,
        'temperature': 0.7,
        'branch_factor': 4,
        'batch_size': 50,
        'batch_interval': 1.0,
        'max_workers_policy': 50,
    },
    
    # Training configuration
    'training': {
        'target_examples_per_question': 40
    },
    
    # File paths configuration
    'paths': {
        'questions_path': 'data_test.txt'
    },
    
    # Intervals configuration
    'intervals': {
        'stats_interval': 60    # 1 minute
    },
    
    # API configuration
    'api': {
        'openai_api_base': "http://79.160.189.79:14161/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': None,  #"http://38.29.145.26:40651/predict"
        'policy_model': "mstojkov/sft-135-checkpoint-3000-improved_policy"
    }
}

def get_config():
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy() 