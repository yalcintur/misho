import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import json
import yaml
import sys
import random
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import set_seed, get_cosine_schedule_with_warmup
from models.valuefunction import ValueFunction  # Import the model defined in valuefunction.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ValueDataset(Dataset):
    """
    Custom dataset for value function training.
    Each example is a tuple: (conversation, score)
    """
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        conversation, score = self.data[idx]
        return conversation, torch.tensor(score, dtype=torch.float32)

    def collate_fn(self, batch):
        conversations, scores = zip(*batch)
        inputs = self.tokenizer.apply_chat_template(
            conversations,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        # If inputs is a dict, use its keys; otherwise, assume it's a tensor of input_ids.
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.stack(scores)
        }

def convert_jsonl_to_train_data(jsonl_file_path: str):
    """
    Converts a JSONL file to a list of (conversation, score) tuples.
    Expects each line to contain a JSON object with keys 'prompt' and 'completion'.
    Only user messages are extracted for the conversation.
    """
    train_data = []
    invalid_count = 0
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        invalid_count += 1
                        continue
                
                # Check for required keys
                if not all(key in data for key in ("prompt", "completion")):
                    invalid_count += 1
                    continue
                
                # Extract only user messages
                user_messages = [msg for msg in data["prompt"] if msg["role"] == "user"]
                if not user_messages:
                    invalid_count += 1
                    continue
                
                # Extract score from the completion content
                completion = data["completion"]
                try:
                    if isinstance(completion, list):
                        content = completion[0].get("content", "")
                    elif isinstance(completion, dict):
                        content = completion.get("content", "")
                    else:
                        content = str(completion)
                    score = float(content)
                    # Optionally enforce score range (e.g., 0.0 to 1.0)
                    if not (0.0 <= score <= 1.0):
                        raise ValueError("Score out of expected range")
                except (ValueError, KeyError, TypeError):
                    invalid_count += 1
                    continue
                
                train_data.append((user_messages, score))
                
            except json.JSONDecodeError:
                invalid_count += 1
                continue
    
    logger.warning(f"Skipped {invalid_count} invalid/malformed entries")
    return train_data

def split_data(data, ratios=(0.8, 0.1, 0.1)):
    """
    Splits data into train, validation, and test sets based on provided ratios.
    The ratios should sum to 1.0.
    """
    assert sum(ratios) == 1.0, "Split ratios must sum to 1.0"
    total = len(data)
    train_size = int(ratios[0] * total)
    val_size = int(ratios[1] * total)
    test_size = total - train_size - val_size
    return random_split(data, [train_size, val_size, test_size])

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataloader and returns the RMSE.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
       for batch in tqdm(dataloader, desc="Evaluating"):
          inputs = {k: v.to(device) for k, v in batch.items()}
          labels = inputs.pop("labels")
          logits = model(**inputs)
          probs = torch.sigmoid(logits)
          loss = F.binary_cross_entropy(probs, labels, reduction='sum')
          total_loss += loss.item()
          total_samples += labels.numel()

    avg_bce_loss = total_loss / total_samples
    return avg_bce_loss


def finetune_value(train_config):
    """
    Fine-tunes the value function model using the provided configuration.
    """
    # Set seeds for reproducibility
    set_seed(train_config["seed"])
    random.seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    
    # Load and preprocess data
    all_data = convert_jsonl_to_train_data(train_config["dataset_file"])
    train_split, val_split, test_split = split_data(all_data, tuple(train_config["split_ratios"]))
    
    # Initialize the model and tokenizer
    model = ValueFunction(train_config["model_name"]).to(train_config["device"])
    tokenizer = model.tokenizer
    
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Create dataset objects and data loaders
    train_dataset = ValueDataset(train_split, tokenizer)
    val_dataset = ValueDataset(val_split, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        collate_fn=val_dataset.collate_fn
    )
    
    base_lr = float(train_config["base_lr"])
    head_lr = float(train_config["head_lr"])
    weight_decay = float(train_config["weight_decay"])

    optimizer = optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': base_lr},
        {'params': model.value_head.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)

    total_steps = len(train_loader) * train_config["max_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config["warmup_steps"],
        num_training_steps=total_steps
    )
    
    global_step = 0
    best_rmse = float('inf')
    patience_counter = 0
    best_folder = None

    for epoch in range(train_config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, batch in progress_bar:
            global_step += 1
            inputs = {k: v.to(train_config["device"]) for k, v in batch.items()}
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            
            if (step + 1) % train_config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Run validation every half epoch
            if global_step % 1200 == 0:
                model.eval()
                val_rmse = evaluate_model(model, val_loader, train_config["device"])
                logger.info(f"Global step {global_step} - Validation RMSE: {val_rmse:.4f}")
                
                # Print results for 20 examples from the validation set
                logger.info("Example predictions from validation set:")
                # Loop over the first 20 examples
                for i in range(min(20, len(val_dataset))):
                    conversation, true_score = val_dataset[i]
                    # Using the model's predict method (ensure evaluation mode)
                    with torch.no_grad():
                        prediction = model.predict(conversation)
                    logger.info(f"Example {i+1}: True Score: {true_score.item():.4f}, Predicted Score: {prediction.item():.4f}")
                    logger.info(f"Conversation: {conversation}")
                
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    patience_counter = 0
                    best_folder = os.path.join(train_config["output_dir"], f"batch_{global_step}")
                    os.makedirs(best_folder, exist_ok=True)
                    # Save the full model (not just state_dict)
                    torch.save(model, os.path.join(best_folder, "full_model.pt"))
                    tokenizer.save_pretrained(best_folder)
                    logger.info(f"Saved full model at global step {global_step}")
                else:
                    patience_counter += 1
                    if patience_counter >= train_config["patience"]:
                        logger.info("Early stopping triggered")
                        break
                model.train()
        
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        if patience_counter >= train_config["patience"]:
            break

    # Final evaluation on test set
    logger.info("Starting final evaluation on test set...")
    if best_folder is not None:
        # Load the best full model
        model = torch.load(os.path.join(best_folder, "full_model.pt"))
    test_dataset = ValueDataset(test_split, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], collate_fn=test_dataset.collate_fn)
    test_rmse = evaluate_model(model, test_loader, train_config["device"])
    logger.info(f"Final Test RMSE: {test_rmse:.4f}")
    
    return model, test_split

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python finetune_value.py <config_file>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device (GPU if available)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = config["train_arguments"]

    finetune_value(train_config)
