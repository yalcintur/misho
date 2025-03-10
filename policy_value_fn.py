import concurrent.futures
from openai import OpenAI
import time
# Set up your client
openai_api_key = ""
openai_api_base = "http://109.198.107.223:51643/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key="sk-placeholder",  # Use a placeholder that looks like a key
)

def policy_fn(question, state, temperature, branch_factor):
    """
    Generate text completions based on query and state using the OpenAI API.
    
    Args:
        question (str): The initial question text
        state (str): The current state of the conversation
        temperature (float): Controls randomness in generation
        branch_factor (int): Number of completions to generate
        
    Returns:
        list: Unique generated text completions
    """
    try:
        # Construct the prompt from question and state
        if state != "":
            question = question + "\n"
        prompt = question + state

        # Attempt to generate completions via API
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

        # Process and format the generated completions
        generated_texts = []
        # Define some defaults:
        result_append_text = " The answer is "  # Example text to append
        state_list = state.split("\n")
        is_final_state = (len(state_list) == 4)
        
        for choice in response.choices:
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                continue  # Skip invalid responses
            text = choice.message.content.strip()
            
            # Format the final text: Only include the generated text, not the prompt
            formatted_text = text
            
            # Add appropriate ending punctuation
            if is_final_state:
                formatted_text += "."
            elif len(state_list) < 4:
                formatted_text += "\n"
                
            generated_texts.append(formatted_text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = [txt for txt in generated_texts if not (txt in seen or seen.add(txt))]
        return unique_texts
        
    except Exception as e:
        print(f"Error in policy_fn: {str(e)}")
        return []  # Return empty list on error

def policy_fn_batch(q_and_states, temperature, branch_factor, max_workers=5):
    """
    Process a batch of (question, state) pairs concurrently, preserving input order.
    
    Args:
        q_and_states (list of tuples): Each tuple is (question, state).
        temperature (float): Temperature parameter for generation.
        branch_factor (int): Number of completions per request.
        max_workers (int): Number of parallel threads.
    
    Returns:
        list of lists: Each inner list contains the unique generated text completions for one input.
    """
    # Prepare a list to store results in the same order as q_and_states.
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
    
    # Return the list of lists directly, with each inner list containing unique responses for one input
    return results

def policy_value_fn(q_and_states, temperature, branch_factor, max_workers=50):
    """
    Process a batch of (question, state) pairs and return a list of lists of actions.
    Each action is prefixed with its corresponding state.
    
    Args:
        q_and_states (list of tuples): Each tuple is (question, state).
        temperature (float): Temperature parameter for generation.
        branch_factor (int): Number of completions per request.
        max_workers (int): Number of parallel threads.
    
    Returns:
        list of lists: Each inner list contains the unique actions for one input,
                      with each action prefixed by its state.
    """
    # Get the raw responses using the policy_fn_batch
    results = policy_fn_batch(q_and_states, temperature, branch_factor, max_workers)
    
    # Process each response to append the state in front of each result
    processed_results = []
    for i, input_responses in enumerate(results):
        actions = []
        _, state = q_and_states[i]  # Get the state for this input
        
        for response in input_responses:
            # Append the state in front of the response
            if state:
                prefixed_response = state + response
            else:
                prefixed_response = response
                
            actions.append((prefixed_response,0.5))
            
        processed_results.append(actions)
    
    return processed_results


#question = "3 7 11 12"
#state = ""


#a = policy_value_fn([(question, state)], temperature=1.0, branch_factor=40)
#for i in a:
#    for idx, item in enumerate(i):
#        print(f"{idx+1}: {item[0]}: {item[1]}")
    print("---")