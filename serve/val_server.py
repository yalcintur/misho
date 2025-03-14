import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

from models.valuefunction import ValueFunction

model = ValueFunction(model_name="/home/yalcintur/Downloads/batch_22800", load_full_model=True)