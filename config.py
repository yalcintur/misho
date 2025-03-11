"""
Configuration for MCTS training.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration
    'forest': {
        'num_trees': 140,
        'max_expansions': 40,
        'max_leaves': 80,
        'c_explore': 0.3,
        'temperature': 0.8,
        'branch_factor': 5,
        'batch_size': 70,
        'batch_interval': 1.0,
        'max_workers_policy': 70,
        'max_workers_value': 30
    },
    
    # Training configuration
    'training': {
        'target_examples_per_question': 140
    },
    
    # File paths configuration
    'paths': {
        'questions_path': 'all_questions.txt',
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
        'openai_api_base': "http://185.185.58.72:40095/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': None,  # Replace with actual URL when ready
        'value_api_endpoint': "/api/value"
    }
}

def get_config():
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy() 