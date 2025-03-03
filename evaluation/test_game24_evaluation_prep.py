import json


data_set_file = "/Users/mihajlostojkovic/Downloads/StanfordU/Winter2025/cs224n/project/misho/test_game24.jsonl"  
data_set_evaluation = "/Users/mihajlostojkovic/Downloads/StanfordU/Winter2025/cs224n/project/misho/evaluation/data_set_evaluation.jsonl"
training_set_evaluation = "/Users/mihajlostojkovic/Downloads/StanfordU/Winter2025/cs224n/project/misho/evaluation/training_set_eval.jsonl"  
training_data_set = "/Users/mihajlostojkovic/Downloads/StanfordU/Winter2025/cs224n/project/misho/train_game24.jsonl"
count = 0 
output_data = {}
with open(data_set_file, "r", encoding="utf-8") as file:
    for line in file:
        entry = json.loads(line)
        user_prompt = entry[0]["content"]
        assistant_response = entry[1]["content"]
        #finds the last sentence of the assistant's response#
        if user_prompt not in output_data:
            output_data[user_prompt] = []
            count += 1
        output_data[user_prompt].append(assistant_response)
        
output_data = [{"input": key, "expected_answers": value} for key, value in output_data.items()]


with open(data_set_evaluation, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4)

print(f"Processed data saved to {data_set_evaluation}")
print(count)
