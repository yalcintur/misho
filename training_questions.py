#!/usr/bin/env python3
import json

def extract_first_line_of_first_user_prompt(filename):
    """
    Reads a JSONL file and extracts the unique first line from the "content"
    of the first dictionary in the "prompt" list (if it exists and has role "user").

    Args:
        filename (str): Path to the JSONL file.
    
    Returns:
        list: A list of unique first-line strings.
    """
    unique_first_lines = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                prompts = record.get("prompt")
                if prompts and isinstance(prompts, list) and len(prompts) > 0:
                    first_message = prompts[0]
                    if first_message.get("role") == "user":
                        content = first_message.get("content", "").strip()
                        if content:
                            first_line = content.splitlines()[0]
                            unique_first_lines.add(first_line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return list(unique_first_lines)

def write_lines_to_file(lines, output_filename):
    """
    Writes each string in lines to the output file, one per line.
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")

if __name__ == '__main__':
    input_filename = "data.jsonl"    # Replace with your input file path
    output_filename = "all_questions.txt"  # Desired output file name

    first_lines = extract_first_line_of_first_user_prompt("/Users/mihajlostojkovic/Downloads/StanfordU/Winter2025/cs224n/project/improved_policy_dataset/train_split.jsonl")
    write_lines_to_file(first_lines, output_filename)
    print(f"Extracted {len(first_lines)} unique first lines to {output_filename}.")
