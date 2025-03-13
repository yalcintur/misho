# valuefunction.py
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
        
       
                self.base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
                hidden_size = self.base_model.config.hidden_size
                self.value_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, 1),
                )
                self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
                
                # Now try to load the state dict
                try:
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        self.load_state_dict(checkpoint)
                except Exception as e:
                    raise ValueError(f"Failed to load model from checkpoint: {e}")
        else:
            # Standard initialization - load base model and create new value head
            print(f"Loading base model: {model_name} and initializing value head")
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            hidden_size = self.base_model.config.hidden_size
            
            # Enhanced value head
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
        
        # Configure tokenizer for right padding (needed for correct last-token extraction)
        if self.tokenizer:
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs):
        return self.base_model.resize_token_embeddings(new_num_tokens, **kwargs)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            
        Returns:
            torch.Tensor: Predicted value (regression output).
        """
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
            
    def prepare_input(self, conversations):
        """
        Convert conversation format to tokenized input for the model
        
        Args:
            conversations: List of conversations in the format
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            Tokenized input ready for the model with input_ids and attention_mask
        """
        # Format conversations
        formatted_texts = []
        for conv in conversations:
            formatted_texts.append(conv)
        
        # Tokenize with the properly configured tokenizer
        tokenized_inputs = self.tokenizer.apply_chat_template(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Handle different return types from tokenizers
        if isinstance(tokenized_inputs, torch.Tensor):
            # If it's a tensor, wrap it in a dictionary and create attention_mask
            input_ids = tokenized_inputs
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif hasattr(tokenized_inputs, 'input_ids'):
            # If it's a BatchEncoding object, convert to dict and ensure attention_mask exists
            input_ids = tokenized_inputs.input_ids
            if hasattr(tokenized_inputs, 'attention_mask'):
                attention_mask = tokenized_inputs.attention_mask
            else:
                attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # If it's already a dictionary, ensure it has attention_mask
        if 'attention_mask' not in tokenized_inputs:
            tokenized_inputs['attention_mask'] = torch.ones_like(tokenized_inputs['input_ids'])
        
        return tokenized_inputs
    
    def predict(self, conversations: List[Dict[str, str]]) -> torch.Tensor:
        """
        Make value predictions for a list of conversations
        
        Args:
            conversations: List containing dictionaries with role and content keys
            
        Returns:
            Value predictions (0-1 probability)
        """
        # Prepare inputs
        inputs = self.prepare_input([conversations])
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            value = self.forward(**inputs)
            value = torch.sigmoid(value)  # Apply sigmoid to get probability
            
        return value