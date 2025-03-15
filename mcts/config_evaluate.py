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
        'temperature': 0.3,
        'branch_factor': 3,
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
        'questions_path': '../data/raw_data/test.txt'
    },
    
    # Intervals configuration
    'intervals': {
        'stats_interval': 60    # 1 minute
    },
    
    # API configuration
    'api': {
        'openai_api_base': "http://136.38.166.236:34733/v1",
        'openai_api_key': "sk-placeholder",
        'value_api_base_url': "http://45.135.56.11:26046/predict",  #"http://45.135.56.11:32637/predict"
        'policy_model': "lakomey/sft-1.7b-base-150-b8" #lakomey/sft-135-iter1-10-b32 lakomey/sft-1.7b-base-150-b8
    }
}

def get_config(value_size: int, policy_size: int, branch_factor: int, num_expansions: int, temperature: float, c_explore: float):
    """
    Get configuration dictionary.
    
    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if value_size == 0:
        config['api']['value_api_base_url'] = None
    if value_size == 135:
        config['api']['value_api_base_url'] = "http://45.135.56.11:26633/predict"
    if value_size == 360:
        config['api']['value_api_base_url'] = "http://45.135.56.11:32637/predict"
    if value_size == 1700:
        config['api']['value_api_base_url'] = "http://45.135.56.11:26046/predict"

    if policy_size == 135:
        config['api']['policy_model'] = "lakomey/sft-135-iter1-10-b32"
        config['api']['openai_api_base'] = "http://81.166.173.12:10569/v1"
    if policy_size == 360:
        config['api']['policy_model'] = "lakomey/sft-360-iter1-50-b8"
        config['api']['openai_api_base'] = "http://79.160.189.79:14182/v1"
    if policy_size == 1700:
        config['api']['policy_model'] = "lakomey/sft-1.7b-base-150-b8"
        config['api']['openai_api_base'] = "http://136.38.166.236:34733/v1"

    config['forest']['branch_factor'] = branch_factor
    config['forest']['max_expansions'] = num_expansions
    config['forest']['temperature'] = temperature
    config['forest']['c_explore'] = c_explore
    return config 