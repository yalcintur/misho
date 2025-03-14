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
from valuefunction import ValueFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ValueDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state_string, score = self.data[idx]
        return state_string, torch.tensor(score, dtype=torch.float32)

    def collate_fn(self, batch):
        states, scores = zip(*batch)
        
        encoded = self.tokenizer(
            list(states),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = torch.stack(scores)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels
        }




def convert_hf_data_to_value_dataset(
    data_path_or_dataset,
    split_name=None
):
    if isinstance(data_path_or_dataset, str):  
        if data_path_or_dataset.endswith(".json") or data_path_or_dataset.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path_or_dataset)
        else:
            dataset = load_from_disk(data_path_or_dataset)
    else:
        dataset = data_path_or_dataset

    if split_name:
        dataset = dataset[split_name]

    data_list = []
    invalid_count = 0

    for example in tqdm(dataset, desc="Converting HF data to ValueDataset"):
        try:
            user_prompt = example.get("prompt", [])
            if not user_prompt or not isinstance(user_prompt, list):
                invalid_count += 1
                continue

            user_text = user_prompt[0].get("content", "").strip()
            if not user_text:
                invalid_count += 1
                continue
            
            # Extract the assistant response (completion)
            assistant_prompt = example.get("completion", [])
            if not assistant_prompt or not isinstance(assistant_prompt, list):
                invalid_count += 1
                continue

            score_str = assistant_prompt[0].get("content", "").strip()
            score_val = float(score_str)  # Convert to float

            data_list.append((user_text, score_val))

        except (ValueError, KeyError, TypeError):
            invalid_count += 1
            continue

    print(f"Skipped {invalid_count} malformed entries.")

    return data_list

# Evaluates the model on a given dataloader and returns the RMSE.
def evaluate_model(model, dataloader, device):
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
    set_seed(train_config["seed"])
    random.seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    
    # Load and preprocess data

    trainer_info = {
    "batch_losses": [],          
    "validation_checkpoints": {} 
    }

    running_loss = 0.0  # accumulates loss for running average
    num_batches = 0 
    
    training_data = convert_hf_data_to_value_dataset(train_config["training_data_file"], "train")
    validation_data = convert_hf_data_to_value_dataset(train_config["validation_data_file"], "train")
    
    model = ValueFunction(train_config["model_name"]).to(train_config["device"])
    tokenizer = model.tokenizer
    
    train_dataset = ValueDataset(training_data, tokenizer)
    validation_dataset = ValueDataset(validation_data, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=train_config["batch_size"],
        collate_fn=validation_dataset.collate_fn
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
    best_loss = float('inf')
    patience_counter = 0
    best_folder = None

    for epoch in range(train_config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, batch in progress_bar:
            global_step += 1
            num_batches += 1
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
            running_loss = epoch_loss / num_batches
            progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            })
            if num_batches % int(len(progress_bar)*train_config["percent_epoch"]//10) == 0:
               print(running_loss)
            


            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else None
            trainer_info["batch_losses"].append({
            "global_step": global_step,
            "batch_index": step,
            "loss": loss.item(),
            "lr": current_lr
             })

            epoch_step = int(len(train_loader) * train_config["percent_epoch"]) #after what percentage of epoch we should evalaute
            # Run validation every tenth epoch
            if global_step %  epoch_step == 0:
                model.eval()
                val_loss = evaluate_model(model, val_loader, train_config["device"])
                logger.info(f"Global step {global_step} - Validation CrossEntropy Loss: {val_loss:.4f}")
                trainer_info["validation_checkpoints"][global_step] = val_loss

        
                # Print results for 20 examples from the validation set
                #logger.info("Example predictions from validation set:")
                # Loop over the first 20 examples
                """for i in range(min(20, len(val_dataset))):
                    conversation, true_score = val_dataset[i]
                    # Using the model's predict method (ensure evaluation mode)
                    with torch.no_grad():
                        prediction = model.predict(conversation)
                    logger.info(f"Example {i+1}: True Score: {true_score.item():.4f}, Predicted Score: {prediction.item():.4f}")
                    logger.info(f"Conversation: {conversation}")"""
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_folder = os.path.join(train_config["output_dir"], f"batch_{global_step}")
                    os.makedirs(best_folder, exist_ok=True)
                    # Save the full model (not just state_dict)
                    torch.save(model, os.path.join(best_folder, "full_model.pt"))
                    tokenizer.save_pretrained(best_folder)
                    checkpoint_info_path = os.path.join(best_folder, "trainer_info.json")
                    with open(checkpoint_info_path, "w") as f:
                        json.dump(trainer_info, f, indent=4)
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
