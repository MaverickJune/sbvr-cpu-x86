import os
import gc
from sbvr_utils.utils_llama import get_llama, sbvr_decompress_on_llama
import torch
import argparse
import sys

def _r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def _g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def _y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def _b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def get_subdirectory_names(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]
    
def convert_name(name):
    parts = name.split('_', 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid name format: {name}")
    return parts[0] + '/' + parts[1].split('_num_sum_')[0]
    
def convert_compressed_sbvr_weights_to_model(compressed_weight_path:str = None,
                                             save_model_path:str = None):
    if None in [compressed_weight_path, save_model_path]:
        raise ValueError("compressed_weight_path and save_dir_path cannot be None")
    compressed_weight_list = get_subdirectory_names(compressed_weight_path)
    print(_g_str(f"Compressed weight list: {compressed_weight_list}"))
    
    for name in compressed_weight_list:
        print(_b_str(f"Processing {name}..."))
        model, _ = get_llama(model_path=convert_name(name), device_map="cuda:0")
        weight_dir_path = os.path.join(compressed_weight_path, name)
        save_dir_path = os.path.join(save_model_path, name)
        print(_y_str(f"Weight dir path: {weight_dir_path}"))
        print(_y_str(f"Save dir path: {save_dir_path}"))
        sbvr_decompress_on_llama(model=model,
                                 weight_dir_path=weight_dir_path,
                                 sbvr_save_path=save_dir_path)
        model = model.to("cpu")
        del model
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        print(_g_str(f"Decompressed model saved to {save_dir_path}"))
        
        weight_dir_path = compressed_weight_path
        save_dir_path = save_model_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed_weight_path", type=str, required=True, default="./compressed_weights",
                        help="Path to the compressed weight directory")
    parser.add_argument("--save_dir_path", type=str, required=True, default="./sbvr_models",
                        help="Path to save the decompressed model")
    args = parser.parse_args()
    
    convert_compressed_sbvr_weights_to_model(
        compressed_weight_path=args.compressed_weight_path,
        save_model_path=args.save_dir_path 
    )
        