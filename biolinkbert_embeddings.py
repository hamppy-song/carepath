# biolinkbert_embeddings.py
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 1. 디바이스 자동 설정 (GPU 있으면 cuda, 없으면 cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 설정
model_name = "michiyasunaga/BioLinkBERT-base"
# 캐시 위치를 바꾸고 싶으면 환경변수 BIOLINKBERT_CACHE_DIR 사용 (없으면 HF 기본 캐시)
cache_dir = os.environ.get("BIOLINKBERT_CACHE_DIR", None)

# 3. 모델 로딩 (Safetensors + 메모리 효율 옵션)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_safetensors=True,
    low_cpu_mem_usage=True,
).to(device)
model.eval()


# 4. BioLinkBERT 임베딩 추출 함수
@torch.no_grad()
def get_biolink_embedding(text, max_length=512):
    """입력 텍스트로부터 BioLinkBERT [CLS] 임베딩 추출."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return embedding


# 5. CLS 토큰 임베딩 추출 함수 (BioBERT-style)
@torch.no_grad()
def get_biolinkbert_cls_embedding(text, max_length=512):
    """CLS 토큰 임베딩을 추출."""
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
