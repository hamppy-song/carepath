# biolinkbert_embeddings.py
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "michiyasunaga/BioLinkBERT-base"

cache_dir = os.environ.get("BIOLINKBERT_CACHE_DIR", None)


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_safetensors=True,
    low_cpu_mem_usage=True,
).to(device)
model.eval()



@torch.no_grad()
def get_biolink_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return embedding


@torch.no_grad()
def get_biolinkbert_cls_embedding(text, max_length=512):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,   # dynamic padding
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding
