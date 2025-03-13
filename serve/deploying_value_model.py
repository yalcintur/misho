# fastapi_app.py
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import uvicorn
from pydantic import BaseModel
import torch
import sys
import torch
import logging

# Define a request model
class RequestData(BaseModel):
    messages: list  # Example: [{"role": "user", "content": "..."}, ...]

app = FastAPI()

# Load your model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/misho/value_models/sft-135_value_iter1/batch_1192/full_model.pt"  # Adjust the path
model = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
tokenizer = model.tokenizer  # Assuming it's saved with the model

@app.post("/predict")
async def predict_value(data: RequestData):
    try:
        # Extract the state from the conversation.
        # We assume the first message contains the prompt/state.
        if not data.messages:
            raise ValueError("Conversation list is empty.")
        state = data.messages[0].get("content", "").strip()
        if not state:
            raise ValueError("No state found in the conversation prompt.")
        
        # Call the model's predict method on the extracted state.
        value = model.predict(state)
        
        value_float = value.item() if isinstance(value, torch.Tensor) else value
        return {"value": value_float}
    except Exception as e:
        print(e)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
