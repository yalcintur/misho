import os
import yaml
import argparse
from finetuning.finetune_policy_improved import finetune_policy
from finetuning.finetune_value import finetune_value
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a YAML file.')
    parser.add_argument('config', type=str, help='Path to the YAML file')
    args = parser.parse_args()

    config = read_yaml(args.config)
    train_config = config['train_arguments']
    
    if train_config["model_function"] == 'policy':
        finetune_policy(train_config)
    elif train_config["model_function"] == 'value':
        finetune_value(train_config)
    