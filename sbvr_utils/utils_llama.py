from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
# from models.sbvr_llama import SBVRLlamaForCausalLM
import torch
import sbvr
import os
import sys

from sbvr_utils.log_config import get_logger
logger = get_logger(__name__)


@torch.inference_mode
def decompress_sbvr_llama(weight_path=None, model=None):
    pass


@torch.no_grad()
def get_llama(model_path="meta-llama/Llama-3.2-3B-Instruct", tokenizer_path="meta-llama/Llama-3.2-3B-Instruct", 
              device_map:str ="auto", use_sbvr:bool = False, use_llm_int8:bool = False, use_fp8:bool = False,
              use_gptq_4:bool = False, use_awq_4:bool = False, load_from_local:bool = False, gptq_local_model_path:str = None,
              weight_path:str = None, sbvr_from_local_path:str = None):
    r'''
    Fetch llama model from huggingfaces

    @param model_path: target model to fetch from huggingface
    @param tokenizer_path: In case you want to use different tokenizer
    '''
    if not tokenizer_path:
        tokenizer_path = model_path
        
    if use_sbvr:
        if sbvr_from_local_path:
            model = LlamaForCausalLM.from_pretrained(
                sbvr_from_local_path,
                torch_dtype=torch.float16,
                device_map=device_map   
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            
            return model, tokenizer
        if weight_path is None:
            raise ValueError("weight_path cannot be None when use_sbvr is True")
        logger.info("Using SBVR Llama model")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        sbvr_decompress_on_llama(model, tokenizer, weight_path, model_path)
    elif use_llm_int8:
        logger.info("Using Llama model with LLM.int8")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    elif use_fp8: # Only works in hopper GPU (compute capability 9.0 and above)
        logger.info("Using Llama model with FP8")
        from transformers import FineGrainedFP8Config
        fp8_config = FineGrainedFP8Config()
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=fp8_config
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    elif use_gptq_4:
        from transformers import GPTQConfig
        logger.info("Using Llama model with GPTQ 4-bit")
        
        if not load_from_local:
            logger.info("Start quantizing to GPTQ model ...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, quantization_config=gptq_config)
            
            model.to(device_map)
            model_path = model_path.replace("meta-llama/", "")
            model.save_pretrained(f"{model_path}-gptq-4bit")
            tokenizer.save_pretrained(f"{model_path}-gptq-4bit")
            logger.info("Finish quantizing GPTQ model")
            logger.info(f"Saved it to the local path: {model_path}-gptq-4bit")
        else:
            if gptq_local_model_path is None:
                raise ValueError("local_model_path cannot be None when load_from_local is True")
            logger.info("Loading local GPTQ model...")
            tokenizer = AutoTokenizer.from_pretrained(f"{gptq_local_model_path}")
            model = AutoModelForCausalLM.from_pretrained(f"{gptq_local_model_path}", device_map=device_map)
            logger.info("Finish loading local GPTQ model")
            
    elif use_awq_4:
        from transformers import __version__ as transformers_version
        if not transformers_version == "4.47.1":
            raise ValueError("AWQ 4-bit quantization only works with transformers version 4.47.1")
        logger.info("Using Llama model with AWQ 4-bit")
        allowed_models = [("Llama-3.2-1B", "joshmiller656/Llama3.2-1B-AWQ-INT4"), 
                          ("Llama-3.1-8B", "solidrust/Hermes-3-Llama-3.1-8B-AWQ")]
        flag = False
        quantized_model_name = ""
        for item in allowed_models:
            if item[0] in model_path:
                flag = True
                quantized_model_name = item[1]
                break
        if not flag:
            raise ValueError(f"Model {model_path} is not supported for AWQ 4-bit. Supported models: {allowed_models}")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_name,
            device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_name, use_fast=False)
            
    else: 
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    model.eval()
    
    return model, tokenizer


@torch.inference_mode()
def sbvr_decompress_on_llama(model, tokenizer=None, weight_dir_path:str=None, model_path:str = None, sbvr_save_path:str = None):
    
    if weight_dir_path is None:
        raise ValueError("weight_path cannot be None")
    
    attn_weights_name = ["q", "k", "v"]
    ffn_weights_name = ["gate_proj", "down_proj", "up_proj"]
 
    for i, layer in enumerate(model.model.layers):
        layer_path = os.path.join(weight_dir_path, f"layer_{i}_")
        for weight_name in attn_weights_name:
            device = layer.self_attn.__getattr__(weight_name + "_proj").weight.device
            weight_path = layer_path + f"{weight_name}.pt"
            sbvr_weight = sbvr.load(weight_path, device=device)
            layer.self_attn.__getattr__(weight_name + "_proj").weight = torch.nn.Parameter(sbvr_weight.decode().to(device), requires_grad=False)
            logger.info(f"Decompressed {weight_name} weight from {weight_path}")
        for weight_name in ffn_weights_name:
            device = layer.mlp.__getattr__(weight_name).weight.device
            weight_path = layer_path + f"{weight_name}.pt"
            sbvr_weight = sbvr.load(weight_path, device=device)
            layer.mlp.__getattr__(weight_name).weight = torch.nn.Parameter(sbvr_weight.decode().to(device), requires_grad=False)
            logger.info(f"Decompressed {weight_name} weight from {weight_path}")
            
    logger.info("Decompression complete")
    model.save_pretrained(sbvr_save_path)
    logger.info(f"Saved decompressed model to {sbvr_save_path}")
    


@torch.no_grad()
def get_layer_ffn_weight(model, layer_idx):
    r"""
    Get the ffn(gate_proj) weight of the decoder layer[layer_idx] from the llama model
    """
    ffn_weight = model.model.layers[layer_idx].mlp.gate_proj.weight
    ffn_weight = ffn_weight.detach().clone()

    return ffn_weight


def format_llama3(input:str = None, tokenizer = None):
    r'''
    Format input into the right llama3 instruct format
    '''

    if None in (input, tokenizer):
        raise ValueError("input or tokenizer should not be None")
    if type(input) != str:
        raise ValueError("input must be a string")
    
    def reformat_llama_prompt(text):
        r"""
        Remove the "Cutting Knowledge Date" and "Today Date" lines from the text. \n
        Add a newline before the "<|start_header_id|>user<|end_header_id|>" marker.
        """
        marker_user = "<|start_header_id|>user<|end_header_id|>"
        marker_assistant = "<|start_header_id|>assistant<|end_header_id|>"

        lines = text.splitlines()
        result = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("Cutting Knowledge Date:"):
                i += 1
                continue
            elif lines[i].startswith("Today Date:"):
                i += 1
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
            else:
                if marker_user in lines[i]:
                    modified_line = lines[i].replace(marker_user, "\n"+marker_user)
                    result.append(modified_line)
                else:
                    result.append(lines[i])
                i += 1
                
        if result:
            result[-1] = result[-1] + marker_assistant
        return "\n".join(result)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible."},
        {"role": "user", "content": input}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    formatted_input = reformat_llama_prompt(formatted_input)

    return formatted_input