from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

local_model_path = "/data/yalcin/sft_output_135-instruct/checkpoint-3000"

model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

custom_eos_token = "."

tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(custom_eos_token)
model.config.eos_token_id = tokenizer.eos_token_id

prompt = "4 4 4 4"

messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, temperature=0.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))