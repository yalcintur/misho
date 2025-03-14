from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

HF_TOKEN = "..." 

MODEL_PATH = "/models/checkpoint-200/"  

REPO_NAME = "lakomey/sft-135-iter1-200" 

login(token=HF_TOKEN)

print(f"Loading model and tokenizer from {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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

with open("README.md", "w") as f:
    f.write(model_card)

print(f"Uploading model and tokenizer to {REPO_NAME}")
model.push_to_hub(REPO_NAME, commit_message="Upload fine-tuned model")
tokenizer.push_to_hub(REPO_NAME, commit_message="Upload tokenizer")

print(f"Model successfully uploaded to: https://huggingface.co/{REPO_NAME}")