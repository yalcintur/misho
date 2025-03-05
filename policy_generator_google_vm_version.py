import json
import torch
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    max_tokens=100,   # Adjust as needed
    temperature=0,    # Deterministic output
    top_p=0.7
)
torch.cuda.empty_cache()
def list_of_states_to_expand(query:str, state:str, model_path:str, n_actions_to_generate=5, temperature=1):
    # Load and update the tokenizer (for chat formatting and custom EOS token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    state_list = state.split("\n")
    if (len(state_list) - 1) < 3:
       stop_index = 25
    else:
       stop_index = 198
    # If the chat template is already configured, no need to call setup_chat_format again.
    # You can, if needed, do: tokenizer = setup_chat_format(model=None, tokenizer=tokenizer)[1]
    sampling_params = SamplingParams(   # Adjust as needed
    temperature=temperature,    # Deterministic output
    top_p=0.95,
    max_tokens = 90,
    stop_token_ids = [stop_index] #index for "}" as end-token

    )
   
   # custom_eos_token = "\n"#
   # tokener.eos_token_id = tokenizer.convert_tokens_to_ids(custom_eos_token)
   
    # Save changes (optional, to update the model directory)
    tokenizer.save_pretrained(model_path)
        # Initialize vLLM using the model directory.
    newline = "\n"
    print(f"Tokenizer tokenize {tokenizer.tokenize(newline)}")

    print(tokenizer.encode(")", add_special_tokens=False))

    llm = LLM(model=model_path, dtype="auto")
    if state != "":
       query = query + "\n" + "assistant\n"

    prompt = query + state
    print(f"This is a prompt within the function: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"this is a formatted string: {formatted_prompt}")
        # Generate multiple actions for the prompt
    for _ in range(n_actions_to_generate):  # Use a proper loop variable
       result = llm.generate([formatted_prompt], sampling_params = sampling_params)
       generated_text = result[0].outputs[0].text.strip()
       print(f"Generated text {generated_text}")
      
# Remove the prefix "assistant\n" if present.
       if generated_text.startswith("assistant\n"):
    	   cleaned_text = generated_text[len("assistant\n"):]
       else:
    	   cleaned_text = generated_text

# Remove any trailing newline characters to check the ending.
       cleaned_text = cleaned_text.rstrip("\n")

# Ensure the text ends with a closing parenthesis.
       if not cleaned_text.endswith(")"):
    	   cleaned_text = cleaned_text + ")"

# Finally, add an end-of-line token.
       cleaned_text = cleaned_text + "\n"

       print(f"Cleaned text: {cleaned_text}")

q = "5 5 7 10"
s= "3 * 5 = 15 (left: 3 12 15)\n12 - 15 = -3 (left: -3 5 7)\n"
full_prompt = q
print(full_prompt)

print(list_of_states_to_expand(q, s, model_path =  "/home/mstojkov2025/project/sft-1.7b-checkpoint-1000"))

            

        



   
