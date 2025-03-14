import csv
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

api_key = "<YOUR_API_KEY>" 
client = OpenAI(api_key=api_key)

def read_numbers_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def ask_gpt4o_for_24_solution(numbers):
    prompt = f"Given these four numbers: {numbers}, find a mathematical expression using +, -, *, / that evaluates to 24."

    # different models like o3-mini can be used here as well
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in solving the 24 game."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def save_results_to_csv(results, output_filename):
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Numbers", "Solution"])
        writer.writerows(results)

def main():
    input_filename = "/home/yalcintur/workspace/courses/misho/data_test.txt"
    output_filename = "solutions.csv"

    number_sets = read_numbers_from_file(input_filename)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(ask_gpt4o_for_24_solution, numbers): numbers for numbers in number_sets}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing requests", unit="req"):
            numbers = futures[future]
            try:
                solution = future.result()
            except Exception as e:
                solution = f"Error: {e}"
            results.append([numbers, solution])

    save_results_to_csv(results, output_filename)
    print(f"Results saved to {output_filename}")

if __name__ == "__main__":
    main()
