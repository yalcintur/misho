import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from trl import setup_chat_format
from typing import List, Dict, Union, Optional

class ValueFunction(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M", dropout_rate=0.1):
        """
        Initialize the value function based on SmolLM2
        
        Args:
            model_name: The name or path of the base model to use
            dropout_rate: Dropout rate for the classification head
        """
        super(ValueFunction, self).__init__()
        
        # Load the base model using AutoModelForCausalLM like in your policy function
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use the same setup_chat_format function as in your policy code
        self.base_model, self.tokenizer = setup_chat_format(
            model=self.base_model, 
            tokenizer=self.tokenizer
        )
        
        # Extract hidden size from the model config
        hidden_size = self.base_model.config.hidden_size
        
        # Create value head (classification layer)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
            
    def prepare_input(self, conversations: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
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
            max_length=1024
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
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the model
        
        Args:
            input_ids: Tokenized input ids
            attention_mask: Attention mask for the input
            
        Returns:
            value: Value prediction (0-1) for each input
        """
        # If attention_mask is None, create one (all 1s)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get the outputs from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get the representation of the last token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        
        # Extract the hidden state for the last token of each sequence
        last_token_hidden_states = torch.stack(
            [last_hidden_state[i, sequence_lengths[i], :] for i in range(batch_size)]
        )
        
        # Apply the value head to get the value prediction
        value = self.value_head(last_token_hidden_states)
        
        return value
    def predict(self, conversations: List[Dict[str, str]]) -> torch.Tensor:
        """
        Make value predictions for a list of conversations
        
        Args:
            conversations: List containing dictionaries with role and content keys
            
        Returns:
            Value predictions between 0 and 1
        """
        # Prepare inputs
        inputs = self.prepare_input([conversations])
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            value = self.forward(**inputs)
            
        return value