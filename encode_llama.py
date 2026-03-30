from transformers import LlamaForCausalLM, AutoTokenizer
from models.sbvr_llama import SBVRLlamaForCausalLM
from sbvr_utils.utils_llama import get_llama
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import sbvr
import os
import argparse

from sbvr_utils.log_config import get_logger
logger = get_logger(__name__)


@torch.no_grad()
def process_single_decoder_layer(layer_idx, target_layer, curr_device, num_sums=4, save_path=None):
    logger.info(f"Processing layer {layer_idx} on GPU {curr_device}...")
    
    attn_weights = [
        ("q", target_layer.self_attn.q_proj.weight),
        ("k", target_layer.self_attn.k_proj.weight),
        ("v", target_layer.self_attn.v_proj.weight),
    ]
    ffn_weights = [
        ("gate_proj", target_layer.mlp.gate_proj.weight),
        ("down_proj", target_layer.mlp.down_proj.weight),
        ("up_proj", target_layer.mlp.up_proj.weight),
    ]
    total_weights = attn_weights + ffn_weights
    
    for weight_name, target_weight in total_weights:
        logger.info(f"Processing {weight_name} weight...")
        weight_path = os.path.join(save_path, f"layer_{layer_idx}_{weight_name}.pt")
        target_weight = target_weight.to(curr_device)
        sbvr_compressed_weight = sbvr.sbvr(target_weight, 
                                           encoder_config={"num_sums": num_sums},
                                           device=curr_device)
        sbvr_compressed_weight.save(weight_path)
        logger.info(f"Saved {weight_name} weight to {weight_path}")
    
@torch.no_grad()
def process_lm_head(lm_head, num_sums=4, curr_device=0, save_path=None):
    logger.info("Processing lm_head...")
    weight_path = os.path.join(save_path, "lm_head_weight.pt")
    lm_head_weight = lm_head.weight.to(curr_device)
    sbvr_compressed_weight = sbvr.sbvr(lm_head_weight, 
                                       encoder_config={"num_sums": num_sums},
                                       device=curr_device)
    sbvr_compressed_weight.save(weight_path)
    logger.info(f"Saved lm_head weight to {weight_path}")
    

@torch.no_grad()
def process_sbvr_llama_multi_gpu(model, num_sums=4, 
                                 compressed_weight_path="compressed_weights",
                                 save_path=None):
    # if save_path is None:
    #     raise ValueError("save_path cannot be None")
    # curr_dir = os.path.dirname(os.path.abspath(__file__))
    # save_path = os.path.join(curr_dir, save_path)
    os.makedirs(compressed_weight_path, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    n_layers = len(model.model.layers)
    n_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {n_gpus}")
    
    if n_gpus == 0:
        raise ValueError("No GPUs available for processing")
    
    curr_device = 0
    proc_list = [None for _ in range(n_gpus)]
    
    logger.info(f"Processing {n_layers} layers across {n_gpus} GPUs")
    
    for layer_idx in range(n_layers):
        if proc_list[curr_device] is not None:
            proc_list[curr_device].join()
        if curr_device + 1 < n_gpus and proc_list[curr_device + 1] is not None:
            proc_list[curr_device + 1].join()
            
        proc_list[curr_device] = mp.Process(
            target=process_single_decoder_layer,
            args=(layer_idx, model.model.layers[layer_idx].cpu(), curr_device, num_sums, compressed_weight_path)
        )
        proc_list[curr_device].start()
        curr_device = (curr_device + 1) % n_gpus
        
    for p in proc_list:
        p.join()
           
    logger.info("Processing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num_sums", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="./compressed_weights")
    parser.add_argument("--compressed_weight_path", type=str, default="./sbvr_models")
    args = parser.parse_args()
    
    # MODEL_PATH = "meta-llama/Llama-3.1-8B"
    # NUM_SUMS = 4
    # SAVE_PATH = f"compressed_weights/{MODEL_PATH.split('/')[-1]}"
    
    model, tokenizer = get_llama(model_path=args.model_path, device_map="cpu")
    process_sbvr_llama_multi_gpu(model, num_sums=args.num_sums, 
                                 compressed_weight_path=args.compressed_weight_path,
                                 save_path=args.save_path)
    