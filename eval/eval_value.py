import torch
import torch.nn as nn
import json
import yaml
import sys
from typing import List, Dict, Tuple
import argparse
from models import ValueFunction

def convert_jsonl_to_train_data(jsonl_file_path: str) -> List[Tuple[List[Dict[str, str]], float]]:
    """
    Converts a JSONL file with prompt-completion pairs into training data.
    Only uses the user's message as input and extracts the score from completion.
    
    Args:
        jsonl_file_path: Path to the JSONL file containing prompt-completion pairs
    
    Returns:
        A list of tuples, where each tuple contains:
            - A list containing only the user's message dictionary
            - A float representing the score/correctness of the completion
    """
    train_data = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if isinstance(data, str):
                    data = json.loads(data)
                
                # Get only the user's message from prompt
                user_messages = [msg for msg in data["prompt"] if msg["role"] == "user"]
                
                # Get the completion message to extract the score
                completion_messages = data["completion"]
                
                # Extract score from completion
                if isinstance(completion_messages, list) and len(completion_messages) > 0:
                    try:
                        score = float(completion_messages[0]["content"])
                    except (ValueError, KeyError):
                        score = 1.0
                elif isinstance(completion_messages, dict):
                    try:
                        score = float(completion_messages["content"])
                    except (ValueError, KeyError):
                        score = 1.0
                else:
                    score = 1.0
                
                # Only append the user messages and score
                train_data.append((user_messages, score))
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                print(f"Problematic line: {line[:100]}...")
                continue
    
    return train_data

def evaluate_model(model, test_data, device, batch_size=32):
    """
    Evaluates the model on test data.
    
    Args:
        model: The value function model
        test_data: List of (messages, score) tuples
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        average loss, accuracy
    """
    model.eval()
    criterion = nn.BCELoss(reduction='mean')
    
    total_loss = 0.0
    correct = 0
    batch_count = 0
    
    # Adjust batch size if needed
    actual_batch_size = min(batch_size, len(test_data))
    
    with torch.no_grad():
        for i in range(0, len(test_data), actual_batch_size):
            batch = test_data[i:i+actual_batch_size]
            conversations = [item[0] for item in batch]
            
            print("Conversations: ", conversations)
            targets = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(device)
            
            # Convert to model inputs
            inputs = model.prepare_input(conversations)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            # No need to apply sigmoid here as it's part of the model
            print(f"Probabilities: {outputs.view(-1)}")
            print(f"Target scores: {targets}")
            
            # Calculate loss
            loss = criterion(outputs.view(-1), targets)
            total_loss += loss.item()
            
            # Use outputs directly for predictions
            predictions = (outputs.view(-1) > 0.5).float()
            correct += (predictions == targets).sum().item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"Processed {batch_count} batches...")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate a value function model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default="/data/yalcin/value/sft_output_135M/smol-lm-135M/model.pt", 
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = config['train_arguments']
    
    # Load and verify model
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = ValueFunction(config['model_name'])  # Initialize with same architecture
        model.load_state_dict(checkpoint)  # Changed: load directly from checkpoint
        if not isinstance(model, ValueFunction):
            raise TypeError(f"Loaded model is of type {type(model)}, expected ValueFunction")
        print("Model loaded successfully")
        print(f"Model architecture: {model.__class__.__name__}")
        # Print model parameters count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model = model.to(device)
    model.eval()  # Ensure model is in evaluation mode
    
    #test_data = convert_jsonl_to_train_data("/home/weh4401/st/24/misho/value_training_data.jsonl")[-20:]
    #print(test_data)
    #test_data = [([{'role': 'user', 'content': '1 3 6 8\n\n'}, {'role': 'assistant', 'content': '0.0'}], 0.0)]
    #test_data = [([{"role": "user", "content": "3 3 10 13\n13-3=10 (left: 3, 10, 10)\n10+10=20 (left: 3, 20)\n20+3=23 (left: 23)\nThe solution is: 13-3+10+3 = 23.\n"}], 0.0)]
    #test_data = [([{"role": "user", "content": "3 3 10 13\n13-3=10 (left: 3, 10, 10)\n10+10=20 (left: 3, 20)\n20+3=23 (left: 23)\nThe solution is: 13-3+10+3 = 23.\n"}], 0.0)]
    test_data = [([{"role": "user", "content": "1 5 10 10\n10+10=20 (left: 1, 5, 20)\n"}], 1.0)]
    #test_data = [([{"role": "user", "content": "3 7 11 12\n3+11=14 (left: 7, 12, 14)"}], 1.0), ([{"role": "user", "content": "3 7 11 12\n3-11=-8 (left: 7, 12, -8)"}], 0.0)]
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate
    print("\nEvaluating model...")
    avg_loss, accuracy = evaluate_model(
        model, 
        test_data, 
        device, 
        batch_size=1
    )
    
    print("\n=== Evaluation Results ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()