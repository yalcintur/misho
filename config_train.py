"""
Configuration for MCTS training.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration
    'forest': {
        'num_trees': 60,
        'max_expansions': 20,
        'c_explore': 0.3,
        'temperature': 0.7,
        'branch_factor': 3,
        'batch_size': 30,
        'batch_interval': 1.0,
        'max_workers_policy': 30,
        'max_workers_value': 30,
    },
    
    # Training configuration
    'training': {
        'target_examples_per_question': 40
    },
    
    # File paths configuration
    'paths': {
        'train_questions_path': 'data_train.txt',
        'val_questions_path': 'data_val.txt',
        'train_policy_data_path': 'policy_training_data.jsonl',
        'val_policy_data_path': 'policy_validation_data.jsonl',
        'train_value_data_path': 'value_training_data.jsonl',
        'val_value_data_path': 'value_validation_data.jsonl',
        'train_stats_path': 'training_stats.json',
        'val_stats_path': 'validation_stats.json'
    },
    
    # Intervals configuration
    'intervals': {
        'export_interval': 120,  # 2 minutes
        'stats_interval': 60    # 1 minute
    },
    
    # API configuration
    'api': {
        'openai_api_base': "http://172.81.127.5:31540/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': "http://38.29.145.26:40651/predict",  #"http://38.29.145.26:40651/predict"
        'policy_model': "lakomey/sft-135-iter1-10-b32"
    }
}

def get_config():
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy() 