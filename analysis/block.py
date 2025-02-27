import os
import torch
from dotenv import load_dotenv
import sys
import argparse
import numpy as np
import importlib
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoProcessor)

from benchmark import BenchmarkBaseline

from src import (ImportModel,
                 LayerUnits,
                 AverageTaskStimuli,
                 LocImportantUnits,
                 ToMLocDataset,
                 ExtendedTomLocGPT4,
                 ZeroingAblation,
                 MeanImputation,
                 Assessment,
                 load_chat_template)
from analysis.utils import dice_coefficient

def set_model(model_checkpoint: str,
              model_type: str,
              token_hf: str,
              cache_dir: str,
              model_func: str="AutoModelForCausalLM"):
    """ Set the class ImportModel. """
    # dynamically use the model loader
    module = importlib.import_module("transformers")
    model_fn = getattr(module, model_func)
    
    # Set the config
    config = AutoConfig.from_pretrained(model_checkpoint, cache_dir=cache_dir, token=token_hf)
    config.tie_word_embeddings = False
    # config.torch_dtype = "bfloat16"
    
    if model_type=="LLM":
        model = model_fn.from_pretrained(model_checkpoint,
                                         torch_dtype="bfloat16",
                                        device_map="auto",
                                        cache_dir=cache_dir,
                                        token=token_hf)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                  cache_dir=cache_dir,
                                                  token=token_hf)
        import_model = ImportModel(model_type=model_type,
                                    model=model,
                                    tokenizer=tokenizer)
    elif model_type == "VLM":
        chat_template = load_chat_template("dataset/save_checkpoints/vlm_chat_template.txt")

        model = model_fn.from_pretrained(model_checkpoint,
                                        device_map="auto",
                                        cache_dir=cache_dir,
                                        token=token_hf)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, 
                                                  cache_dir=cache_dir, 
                                                  token=token_hf)
        
        processor = AutoProcessor.from_pretrained(model_checkpoint, 
                                                cache_dir=cache_dir, 
                                                token=token_hf)
        import_model = ImportModel(model_type,
                                   model,
                                   tokenizer,
                                   processor,
                                   chat_template)
    else:
        raise ValueError(f"{model_type} is not a model architecture supported.")

    return import_model







        


        






        




