import json
import os

# Define the parent directory to process
parent_directory = "/home/yalcintur/workspace/courses/misho/data/iter2_135_ds"

# Iterate through all subdirectories
for root, _, files in os.walk(parent_directory):
    for file in files:
        if file.endswith(".jsonl"):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, f"{os.path.splitext(file)[0]}-f.jsonl")

            try:
                with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
                    for line in infile:
                        # First, parse the stringified JSON object
                        first_decode = json.loads(line.strip())
                        decoded_json = json.loads(first_decode)

                        # Extract the text content
                        prompt_text = decoded_json["prompt"]
                        completion_text = decoded_json["completion"]

                        # Create a new properly formatted JSON object
                        fixed_json = {"prompt": prompt_text, "completion": completion_text}

                        # Write back as valid JSONL
                        outfile.write(json.dumps(fixed_json) + "\n")

                print(f"Fixed JSONL saved as: {output_file}")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")
