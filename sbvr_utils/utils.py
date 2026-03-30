import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import random
from tqdm import tqdm

from sbvr_utils.log_config import get_logger, ExtLogger
logger = get_logger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
@torch.no_grad()   
def eval_ppl(model=None, tokenizer=None, dataset="wikitext-2", seqlen=2048, n_samples=-1):
    """
    Evaluate the perplexity of the model on a dataset.
    
    @param model: The model to evaluate.
    @param dataset: The dataset to evaluate on.
    @param seqlen: The sequence length for evaluation.
    @param n_samples: The number of samples to evaluate.
    """
    SUPPORTED_DATASETS = ["wikitext-2"]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset} is not supported. Supported datasets are: {SUPPORTED_DATASETS}")
    if None in (model, tokenizer):
        raise ValueError("model and tokenizer cannot be None")
    
    model.eval()
    
    # if dataset == "wikitext-2":
    #     testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #     testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt", truncation=False,).input_ids
    #     logger.info(f"Size of test data: {testenc.numel() * testenc.element_size() / 1024 / 1024 / 1024:.3f} GB")
        
    #     max_samples = testenc.numel() // seqlen
    #     testenc = testenc[0, :(seqlen * max_samples)].view(max_samples, -1)
    #     if n_samples != -1:
    #         testenc = testenc[:n_samples]
    #     else:
    #         n_samples = max_samples
    #     logger.info(f"Number of samples: {n_samples}")
    # else:
    #     raise NotImplementedError(f"Dataset {dataset} is not implemented for now")
        
    # loss_fct = torch.nn.CrossEntropyLoss().cuda()
    # acc_loss = 0.0
    # progress = tqdm(range(n_samples), desc="Evaluating", unit="sample", ncols=80)
    # for i in progress:
    #     input = testenc[i, :].cuda().view(1, -1)
    #     output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
    #     shift_logits = output[:, :-1, :].contiguous()
    #     shift_labels = input[:, 1:]
    #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #     acc_loss += loss.item()
    #     progress.set_description(f"avg_loss = {acc_loss/(i+1):.4f}")
    # acc_loss = acc_loss / n_samples
    
    # ppl = torch.exp(torch.tensor(acc_loss)).item()
    # logger.info(f"Perplexity: {ppl:.4f}")
    
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
            
@torch.inference_mode()
def save_hidden_vector(model=None, tokenizer=None, dataset='wikitext-2', seqlen=256, n_samples=-1, save_path=None):
    pass