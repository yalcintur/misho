import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

class ValueFunction(nn.Module):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M", dropout_rate: float = 0.1, 
                 load_full_model: bool = False):
        """
        Initialize the value function model.

        Args:
            model_name (str): Pretrained model name or path.
            dropout_rate (float): Dropout probability for the value head.
            load_full_model (bool): If True, loads the entire model from checkpoint instead of 
                                   initializing base model and value head separately.
        """
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.base_model.config.hidden_size
        
        # Define value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1),
        )
        
        # Initialize weights for the value head layers
        nn.init.kaiming_normal_(self.value_head[0].weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.value_head[0].bias)
        nn.init.kaiming_normal_(self.value_head[-1].weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.value_head[-1].bias)
        
        if self.tokenizer:
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs):
        return self.base_model.resize_token_embeddings(new_num_tokens, **kwargs)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get the last hidden state from the base model
        last_hidden = outputs.hidden_states[-1]
        
        # Determine the position of the last non-padded token
        seq_lens = attention_mask.sum(dim=1) - 1 if attention_mask is not None else torch.tensor([input_ids.shape[-1]-1]*input_ids.shape[0])
        
        # Gather the hidden state corresponding to the last token of each sequence
        pooled = last_hidden[torch.arange(last_hidden.size(0)), seq_lens]
        
        return self.value_head(pooled).squeeze(-1)
            
    def prepare_input(self, state):
        # Tokenize with the properly configured tokenizer
        # Ensure we return a dictionary with input_ids and attention_mask
        tokenized_inputs = self.tokenizer(
            state,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        return tokenized_inputs
        
    def predict(self, state : str) -> torch.Tensor:
        inputs = self.prepare_input(state)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
    
    # Make prediction
        with torch.no_grad():
            value = self.forward(**inputs)
            value = torch.sigmoid(value)  # Apply sigmoid to get probability
        return value
