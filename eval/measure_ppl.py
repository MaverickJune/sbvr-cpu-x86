from sbvr_utils.utils import eval_ppl
from sbvr_utils.utils_llama import get_llama


def measure_llama_ppl(model_path, use_sbvr=False, use_llm_int8=False, use_fp8=False, 
                      use_gptq_4=False, use_awq_4=False, gptq_local_model_path:str = None,
                      weight_path=None, sbvr_from_local_path:str = None):
    if not model_path:
        raise ValueError("model_path  cannot be None")
    
    if use_sbvr:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_sbvr=True, weight_path=weight_path, sbvr_from_local_path=sbvr_from_local_path)
    elif use_llm_int8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_llm_int8=True)
    elif use_fp8:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_fp8=True)
    elif use_gptq_4:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_gptq_4=True, load_from_local=True, gptq_local_model_path=gptq_local_model_path)
    elif use_awq_4:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0", use_awq_4=True)
    else:
        model, tokenizer = get_llama(model_path=model_path, device_map="cuda:0")
    eval_ppl(model=model, tokenizer=tokenizer, dataset="wikitext-2")
    
    
if __name__ == "__main__":
    MODEL_PATH = "meta-llama/Llama-3.2-1B"
    WEIGHT_PATH = "/home/nxc/sbvr/compressed_weights"
    GPTQ_LOCAL_MODEL_PATH = "/home/nxclab/wonjun/bvq/Llama-3.2-1B-gptq-4bit"
    SBVR_LOCAL_PATH = "/home/nxclab/wonjun/bvq/sbvr_models/meta-llama_Llama-3.2-1B_num_sum_2"
    
    # measure_llama_ppl(model_path=MODEL_PATH)
    # measure_llama_ppl(model_path=MODEL_PATH, use_llm_int8=True)
    # measure_llama_ppl(model_path=MODEL_PATH, use_gptq_4=True, gptq_local_model_path=GPTQ_LOCAL_MODEL_PATH)
    # measure_llama_ppl(model_path=MODEL_PATH, use_awq_4=True)
    measure_llama_ppl(model_path=MODEL_PATH, use_sbvr=True, weight_path=WEIGHT_PATH, sbvr_from_local_path=SBVR_LOCAL_PATH)

    