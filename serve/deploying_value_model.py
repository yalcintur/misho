from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch

class RequestData(BaseModel):
    messages: list  # Example: [{"role": "user", "content": "..."}, ...]

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your checkpoint
checkpoint_path = "/home/misho/value_models/sft-135_value_iter1/batch_1192/full_model.pt"

model = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.half()
model.to(device)
model.eval()
model = torch.compile(model)

tokenizer = model.tokenizer

@app.post("/predict")
async def predict_value(data: RequestData):
    try:
        if not data.messages:
            raise ValueError("Conversation list is empty.")
        state = [data.messages[i].get("content", "").strip() for i, bb in enumerate(data.messages)]
        if not state:
            raise ValueError("No state found in the conversation prompt.")

        with torch.inference_mode():
            values = model.predict(state)

        value_float = [value.item() if isinstance(value, torch.Tensor) else value for value in values]
        return {"value": value_float}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)