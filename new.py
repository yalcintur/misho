from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

# Set your Hugging Face token
# You can also set it as an environment variable: os.environ["HF_TOKEN"] = "your_token"
HF_TOKEN = "..."  # Replace with your actual token

# Path to your fine-tuned model
MODEL_PATH = "./sft-135-checkpoint-3000/"  # Replace with your model path

# Repository name on Hugging Face (format: username/model-name)
REPO_NAME = "lakomey/sft-135-checkpoint-3000"  # Replace with your desired repo name

# Login to Hugging Face
login(token=HF_TOKEN)

# Load model and tokenizer
print(f"Loading model and tokenizer from {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create model card content
model_card = """
---
language: en
license: mit
tags:
- text-generation
- causal-lm
- your-custom-tag
---

# Model Name

This is a fine-tuned version of [base model] for [specific task].

## Model Details

* **Base Model:** [base model name]
* **Training Data:** [brief description]
* **Training Procedure:** [brief description]
* **Parameters:** [number]
* **Training Time:** [duration]

## Intended Use

[Describe what this model is good for]

## Performance

[Share key metrics and evaluation results]

## Limitations

[Describe known limitations]

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

# Save model card
with open("README.md", "w") as f:
    f.write(model_card)

# Push model, tokenizer, and model card to Hub
print(f"Uploading model and tokenizer to {REPO_NAME}")
model.push_to_hub(REPO_NAME, commit_message="Upload fine-tuned model")
tokenizer.push_to_hub(REPO_NAME, commit_message="Upload tokenizer")

# The model card will be uploaded automatically if in the same directory,
# but we could also upload it explicitly:
# from huggingface_hub import upload_file
# upload_file("README.md", path_in_repo="README.md", repo_id=REPO_NAME)

print(f"Model successfully uploaded to: https://huggingface.co/{REPO_NAME}")