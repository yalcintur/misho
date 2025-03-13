import json

input_file = "/home/yalcintur/Downloads/policy_training_data-iter1.jsonl"
output_file = "/home/yalcintur/Downloads/policy_training_data-iter1-f.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # First, parse the stringified JSON object
        first_decode = json.loads(line.strip())
        decoded_json = json.loads(first_decode)

        # Extract the text content from lists
        prompt_text = decoded_json["prompt"]
        completion_text = decoded_json["completion"]

        # Create a new properly formatted JSON object
        fixed_json = {"prompt": prompt_text, "completion": completion_text}

        # Write back as valid JSONL
        outfile.write(json.dumps(fixed_json) + "\n")

print(f"Fixed JSONL saved as: {output_file}")
