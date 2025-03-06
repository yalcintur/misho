import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from models import ValueFunction

def train_model(train_config, model, train_data, dev_data=None):
    
    model_name = train_config["model_name"]
    save_model_name = train_config["save_model_name"]
    output_dir = train_config["output_dir"]
    max_epochs = int(train_config["max_epochs"])
    batch_size = int(train_config["per_device_train_batch_size"])
    learning_rate = float(train_config["learning_rate"])
    logging_steps = int(train_config["logging_steps"])
    save_steps = int(train_config["save_steps"])
    evaluation_strategy = train_config["evaluation_strategy"]
    eval_steps = int(train_config["eval_steps"])
    use_mps_device = train_config["use_mps_device"]
    dataset_file = train_config["dataset_file"]
    device = train_config["device"]
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.BCELoss(reduction='mean')
    
    # Adjust batch size if dataset is smaller than the batch size
    actual_batch_size = min(batch_size, len(train_data))
    if actual_batch_size < batch_size:
        print(f"Warning: Dataset size ({len(train_data)}) is smaller than requested batch size ({batch_size}). Using batch size of {actual_batch_size} instead.")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Process in batches
        for i in range(0, len(train_data), actual_batch_size):
            batch = train_data[i:i+actual_batch_size]
            conversations = [item[0] for item in batch]
            targets = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(device)
            
            # Prepare batch inputs
            batch_inputs = []
            for conv in conversations:
                batch_inputs.append(conv)
            
            # Convert to model inputs
            inputs = model.prepare_input(batch_inputs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Ensure compatible dimensions for loss calculation
            # Use view to preserve batch dimension even for single samples
            outputs_reshaped = outputs.view(-1)
            
            # Calculate loss with shape-compatible tensors
            loss = criterion(outputs_reshaped, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Print epoch results
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on dev set if provided
        if dev_data:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                val_batch_count = 0
                
                # Adjust validation batch size if needed
                actual_val_batch_size = min(batch_size, len(dev_data))
                
                for i in range(0, len(dev_data), actual_val_batch_size):
                    batch = dev_data[i:i+actual_val_batch_size]
                    conversations = [item[0] for item in batch]
                    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(device)
                    
                    # Prepare batch
                    batch_inputs = []
                    for conv in conversations:
                        batch_inputs.append(conv)
                    
                    # Convert to model inputs
                    inputs = model.prepare_input(batch_inputs)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = model(**inputs)
                    
                    # Ensure compatible dimensions for validation
                    outputs_reshaped = outputs.view(-1)
                    val_loss += criterion(outputs_reshaped, targets).item()
                    
                    # Calculate accuracy with shape-compatible predictions
                    predictions = (outputs_reshaped > 0.5).float()
                    correct += (predictions == targets).sum().item()
                    val_batch_count += 1
                
                # Calculate validation metrics
                avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
                accuracy = correct / len(dev_data) if len(dev_data) > 0 else 0
                print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model

# Example usage
def example_usage():
    # Create model
    model = ValueFunction("HuggingFaceTB/SmolLM2-135M")
    
    # Example training data (format: conversation, label)
    train_data = [
        (
            [
                {"role": "user", "content": "3 3 5 12"},
                {"role": "assistant", "content": "5 - 12 = -7 (left: -7 3 3)."}
            ], 
            1.0  # Correct solution
        ),
        (
            [
                {"role": "user", "content": "3 3 5 12"},
                {"role": "assistant", "content": "5 - 12 = -5 (left: -7 3 2)."}
            ], 
            1.0  # Incorrect solution
        ),
        # Add more examples as needed
    ]
    
    # Train model
    trained_model = train_model(model, train_data, epochs=50)
    
    # Example prediction
    test_conversation = [
        {"role": "user", "content": "4 7 8 9"},
        {"role": "assistant", "content": "8 + 4 = 12 (left: 12 7 9)\n12 * 7 = 84 (left: 84 9)\n84 / 9 = 24\nThe answer is (8 + 4) * 7 / 9 = 24."}
    ]
    
    prediction = trained_model.predict(test_conversation)
    print(f"Value prediction: {prediction.item():.4f}")

if __name__ == "__main__":
    example_usage()