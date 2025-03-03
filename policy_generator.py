import json
import torch
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from vllm import LLM, SamplingParams


def list_of_states_to_expand(states: list[tuple[str, str, int]], model_path:str, n_actions_to_generate = 20):
    # Load and update the tokenizer (for chat formatting and custom EOS token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # If the chat template is already configured, no need to call setup_chat_format again.
    # You can, if needed, do: tokenizer = setup_chat_format(model=None, tokenizer=tokenizer)[1]
    sampling_params = SamplingParams(
    temperature=0.7  
   )
    custom_eos_token = "\n"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(custom_eos_token)
    
    # Save changes (optional, to update the model directory)
    tokenizer.save_pretrained(model_path)
    
    # Initialize vLLM using the model directory.
    llm = LLM(model=model_path, dtype="auto")
    
    trees_actions = {}  # Initialize outside to accumulate results for all states

    for state in states:
        output_actions = set()
        query, partial_state, index = state
        trees_actions[index] = {}
        prompt = query + partial_state
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # Generate multiple actions for the prompt
        for _ in range(n_actions_to_generate):  # Use a proper loop variable
            result = llm.generate([formatted_prompt], sampling_params=sampling_params)
            generated_text = result[0].outputs[0].text.strip()
            output_actions.add(generated_text)
        # Store the unique generated actions in a list
        trees_actions[index][prompt] = list(output_actions)

                


            

        



   