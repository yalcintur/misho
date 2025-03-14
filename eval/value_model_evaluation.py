import torch
from transformers import AutoTokenizer
from models.valuefunction import ValueFunction
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging
import sys
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
class ValueDataset(Dataset):
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
                
                if not all(key in data for key in ("prompt", "completion")):
                    invalid_count += 1
                    continue
                
                user_messages = [msg for msg in data["prompt"] if msg["role"] == "user"]
                if not user_messages:
                    invalid_count += 1
                    continue
                
                completion = data["completion"]
                try:
                    if isinstance(completion, list):
                        content = completion[0].get("content", "")
                    elif isinstance(completion, dict):
                        content = completion.get("content", "")
                    else:
                        content = str(completion)
                    score = float(content)
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

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataloader and returns:
    - Binary Cross-Entropy (BCE) Loss
    - L1 Loss (Mean Absolute Error)
    """
    model.eval()
    total_bce_loss = 0.0
    total_l1_loss = 0.0
    total_samples = 0
    l1_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            logits = model(**inputs)
            probs = torch.sigmoid(logits)

            bce_loss = F.binary_cross_entropy(probs, labels, reduction='sum')
            total_bce_loss += bce_loss.item()

            batch_l1 = torch.abs(probs - labels).view(-1).cpu()
            l1_losses.extend(batch_l1.tolist())
            
            total_l1_loss += batch_l1.sum().item()

            total_samples += labels.numel()

    avg_bce_loss = total_bce_loss / total_samples
    avg_l1_loss = total_l1_loss / total_samples
    percentile_99_l1 = [np.percentile(l1_losses, i) for i in range (85, 100)]

    print(f"Evaluation Results: BCE Loss = {avg_bce_loss:.4f}, L1 Loss = {avg_l1_loss:.4f},  Percentiles: {percentile_99_l1}")

    return avg_bce_loss, avg_l1_loss

def main():
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
     )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "/home/135m/batch_22800/full_model.pt"
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()


    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded and set to evaluation mode.")
    test_conversation_scores_pairs = convert_jsonl_to_train_data("/home/misho/value_training_data_v2.jsonl") #path to json format dataset

    test_dataset = ValueDataset(test_conversation_scores_pairs, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        collate_fn=test_dataset.collate_fn
    )

    print(evaluate_model(model, test_loader, device = "cuda"))
    
    

if __name__ == "__main__":
    main()
