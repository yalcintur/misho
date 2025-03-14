import csv
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

api_key = "<YOUR_API_KEY>" 
client = OpenAI(api_key=api_key)

def read_csv_file(input_filename):
    data = []
    with open(input_filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                data.append((row[0].strip(), row[1].strip()))
    return data

def evaluate_solution_with_gpt4o(question, solution):
    prompt = f"""
    You are an expert at evaluating "24 Game" solutions. The "24 Game" requires using the given four numbers with basic arithmetic operations (+, -, *, /) to make 24.

    Given numbers: {question}
    User's equation: {solution}

    Evaluate if the equation:
    1. Uses ONLY the given numbers (no extra or missing numbers).
    2. Maintains the correct order and count of numbers.
    3. Correctly computes to 24.

    Respond ONLY with:
    - "1" (if the equation is valid and evaluates to 24)
    - "0" (if the equation is incorrect due to wrong result, incorrect numbers, or syntax issues)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating 24 game solutions."},
                {"role": "user", "content": prompt},
            ]
        )
        evaluation = response.choices[0].message.content.strip()
        return "1" if evaluation == "1" else "0"
    except Exception as e:
        return "0"

def save_results_to_csv(results, output_filename):
    """Saves the evaluation results to a CSV file."""
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Numbers", "Solution", "Correctness"])  # Header
        writer.writerows(results)

def main():
    input_filename = "solutions.csv"
    output_filename = "evaluated_solutions.csv"

    data = read_csv_file(input_filename)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_solution_with_gpt4o, question, solution): (question, solution) for question, solution in data}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating solutions", unit="eval"):
            question, solution = futures[future]
            try:
                correctness = future.result()
            except Exception as e:
                correctness = "0" 
            results.append([question, solution, correctness])

    save_results_to_csv(results, output_filename)
    print(f"Results saved to {output_filename}")

    count_zeros = sum(1 for row in results if row[2] == "0")
    count_ones = sum(1 for row in results if row[2] == "1")

    print(f"Total Incorrect (0s): {count_zeros}")
    print(f"Total Correct (1s): {count_ones}")

if __name__ == "__main__":
    main()