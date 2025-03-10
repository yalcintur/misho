"""
Configuration for MCTS training.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Forest configuration
    'forest': {
        'num_trees': 100,
        'max_expansions': 40,
        'max_leaves': 100,
        'c_explore': 0.3,
        'temperature': 0.8,
        'branch_factor': 5,
        'batch_size': 50,
        'batch_interval': 0.1
    },
    
    # Training configuration
    'training': {
        'target_examples_per_question': 100
    },
    
    # File paths configuration
    'paths': {
        'checkpoint_path': 'forest_checkpoint.pkl',
        'policy_data_path': 'policy_training_data.jsonl',
        'value_data_path': 'value_training_data.jsonl',
        'stats_path': 'training_stats.json'
    },
    
    # Intervals configuration
    'intervals': {
        'save_interval': 300,  # 5 minutes
        'export_interval': 60,  # 1 minute
        'stats_interval': 30    # 30 seconds
    }
}

def get_config():
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    return DEFAULT_CONFIG.copy() 