import json
from typing import List, Dict, Tuple, Any

def convert_jsonl_to_train_data(jsonl_file_path: str) -> List[Tuple[List[Dict[str, str]], float]]:
    """
    Converts a JSONL file with prompt-completion pairs into a format compatible with the train_data structure.
    Handles cases where lines might be escaped JSON strings.
    
    Args:
        jsonl_file_path: Path to the JSONL file containing prompt-completion pairs
    
    Returns:
        A list of tuples, where each tuple contains:
            - A list of message dictionaries with "role" and "content" keys
            - A float representing the score/correctness of the completion
    """
    train_data = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                # Try to parse the line as JSON
                data = json.loads(line)
                
                # If the data is still a string (double-encoded JSON), parse it again
                if isinstance(data, str):
                    data = json.loads(data)
                
                # Get the prompt messages
                prompt_messages = data["prompt"]
                
                # Get the completion message
                completion_messages = data["completion"]
                
                # Combine prompt and completion messages
                messages = prompt_messages.copy()
                if isinstance(completion_messages, list) and len(completion_messages) > 0:
                    # Add only the completion message to the list
                    messages.append(completion_messages[0])
                elif isinstance(completion_messages, dict):
                    # Handle case where completion might be a dict directly
                    messages.append(completion_messages)
                
                # Extract the score from the completion content
                try:
                    if isinstance(completion_messages, list) and len(completion_messages) > 0:
                        score = float(completion_messages[0]["content"])
                    elif isinstance(completion_messages, dict):
                        score = float(completion_messages["content"])
                    else:
                        score = 1.0  # Default if format unexpected
                except (ValueError, KeyError):
                    # If content is not directly convertible to float, provide a default
                    score = 1.0  # Default score, adjust as needed
                
                train_data.append((messages, score))
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                print(f"Problematic line: {line[:100]}...")  # Print first 100 chars
                continue  # Skip this line and continue with the next one
    
    return train_data
# Example usage
if __name__ == "__main__":
    # Path to your JSONL file
    jsonl_file = "/home/weh4401/st/24/misho/value_training_data.jsonl"
    
    # Convert the data
    train_data = convert_jsonl_to_train_data(jsonl_file)
    
    # Print the first example to verify the conversion
    if train_data:
        print("First converted example:")
        for message in train_data[0][0]:
            print(f"  {message['role']}: {message['content']}")
        print(f"Score: {train_data[0][1]}")
        
    # Example of how to use in your format
    print("\nHow to use this data:")
    print("train_data = [")
    for example in train_data[:2]:  # Print the first 2 examples
        print("    (")
        print("        [")
        for msg in example[0]:
            print(f"            {msg},")
        print("        ],")
        print(f"        {example[1]}  # Score")
        print("    ),")
    print("    # More examples...")
    print("]")
