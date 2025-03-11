import concurrent.futures
from openai import OpenAI

# Set up your client
openai_api_key = ""
openai_api_base = "http://193.143.121.114:39096/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key="sk-placeholder",  # Use a placeholder that looks like a key
)

def policy_fn(question, state, temperature, branch_factor):
    """
    Generate new states (actions) based on a given question and state using the OpenAI API.
    
    Each new state is constructed by concatenating the prompt (question + state) 
    with the generated output. If a generated output is the last action, a period is appended.
    
    Args:
        question (str): The initial question text.
        state (str): The current state of the conversation.
        temperature (float): Controls randomness in generation.
        branch_factor (int): Number of completions (actions) to generate.
        
    Returns:
        list: Unique new states (actions) generated.
    """
    try:
        # Construct the prompt from question and state.
        # If state is non-empty, add a newline to separate question from state.
        if state != "":
            question = question + "\n"
        prompt = question + state

        # Attempt to generate completions via API.
        try:
            response = client.chat.completions.create(
                model="mstojkov/sft-135-checkpoint-3000-improved_policy",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=branch_factor,
                max_completion_tokens=90,
                stop=[".", "\n"]
            )
        except Exception as e:
            print("Connection error.")
            print(e)
            return []  # Return empty list immediately if the API call fails

        # Process and format the generated completions into new states.
        new_states = []
        # Optionally, you could define some text to append; here we simply concatenate prompt and output.
        state_list = state.split("\n")
        
        # Enumerate the choices so we know which one is last.
        for i, choice in enumerate(response.choices):
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                continue  # Skip invalid responses
            output_text = choice.message.content.strip()
            
            # Create the new state by concatenating the original prompt and the output.
            new_state = prompt + "\n" + output_text
            
            # If this is the last action, ensure its last line ends with a period.
            if i == len(response.choices) - 1:
                new_state = new_state.rstrip()  # Remove trailing whitespace/newlines.
                if not new_state.endswith("."):
                    new_state += "."
            else:
                if not new_state.endswith("\n"):
                    new_state += "\n"
                    
            new_states.append(new_state)
        
        # Remove duplicates while preserving order.
        seen = set()
        unique_states = [s for s in new_states if not (s in seen or seen.add(s))]
        return unique_states
        
    except Exception as e:
        print(f"Error in policy_fn: {str(e)}")
        return []  # Return empty list on error

def policy_fn_batch(q_and_states, temperature, branch_factor, max_workers=5):
    """
    Process a batch of (question, state) pairs concurrently, preserving input order.
    
    Each (question, state) pair is processed by policy_fn, which returns a list of new states.
    The function returns a list of lists, where each sublist corresponds to the new states for the input.
    
    Args:
        q_and_states (list of tuples): Each tuple is (question, state).
        temperature (float): Temperature parameter for generation.
        branch_factor (int): Number of completions (actions) per request.
        max_workers (int): Number of parallel threads.
    
    Returns:
        list of lists: Each sublist contains the unique new states (actions) for the corresponding input.
    """
    # Preallocate a results list in the same order as q_and_states.
    results = [None] * len(q_and_states)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each submitted task to its index.
        future_to_index = {
            executor.submit(policy_fn, q, s, temperature, branch_factor): i 
            for i, (q, s) in enumerate(q_and_states)
        }
        
        # Process tasks as they complete.
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing input at index {idx}: {e}")
                results[idx] = []
    
    return results

# Example usage:
q_and_states = [
    (
        "5 5 7 10", 
        "5+10=15 (left: 5, 7, 15)\n7+15=22 (left: 5, 22)\n22+5=27 (left: 27)"
    ),
    (
        "8 3 4", 
        "8+3=11 (left: 8, 3, 11)\n11+4=15 (left: 15)"
    )
]
temperature = 0.8
branch_factor = 3

all_new_states = policy_fn_batch(q_and_states, temperature, branch_factor)
for i, actions in enumerate(all_new_states):
    print(f"New states for input {i}:")
    for action in actions:
        print(f"  - {action}")
