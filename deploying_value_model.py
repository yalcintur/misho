# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from models.valuefunction import ValueFunction  # Your custom model
from transformers import AutoTokenizer
import uvicorn

# Define a request model
class RequestData(BaseModel):
    conversation: list  # Example: [{"role": "user", "content": "..."}, ...]

app = FastAPI()

# Load your model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/135m/batch_22800/full_model.pt"  # Adjust the path
model = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
tokenizer = model.tokenizer  # Assuming it's saved with the model

@app.post("/predict")
async def predict_value(data: RequestData):
    try:
        # Prepare inputs using your model's predict method
        value = model.predict(data.conversation)
        value_float = value.item() if isinstance(value, torch.Tensor) else value
        return {"value": value_float}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
