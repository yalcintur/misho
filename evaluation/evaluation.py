import json
from vllm import LLM, SamplingParams
from difflib import SequenceMatcher




# Paths to your files
json_file_path = "/home/mstojkov2025/project/misho/misho/evaluation/data_set_evaluation.jsonl"  
model_path = "/home/mstojkov2025/project/misho/checkpoint-700"  

# Load the dataset
with open(json_file_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize vLLM with your model
llm = LLM(model=model_path)

# Set sampling parameters
sampling_params = SamplingParams(
    max_tokens=100,   # Adjust based on expected response length
    temperature=0,    # Set to 0 for deterministic outputs
    top_p=0.95       # Use nucleus sampling
)

# Helper function to compute similarity between two strings
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Store results
correct = 0
total = 0

# Generate responses and compare
for entry in dataset:
    prompt = entry["input"]
    expected_answers = entry["expected_answers"]

    # Generate model output
    result = llm.generate([prompt], sampling_params=sampling_params)
    generated_text = result[0].outputs[0].text.strip()

    # Check if generated text matches any expected answer
    best_match = max(similarity(generated_text, ans) for ans in expected_answers)

    # Define a similarity threshold for correctness
    if best_match == 1:  # Adjust as needed
        correct += 1

    total += 1
    print(f"Prompt: {prompt}\nGenerated: {generated_text}\nBest Match: {best_match:.2f}\n")

# Calculate accuracy
accuracy = correct / total * 100
print(f"Model Accuracy: {accuracy:.2f}%")

