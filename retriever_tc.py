import torch
import numpy as np
from transformers import  AutoModel, AutoTokenizer
import os, re, random, faiss, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

MODEL_PATH = "/work/LG/2025/tc_finetuned.pth"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL)
    def forward(self, ids, msk):
        x = self.encoder(ids, attention_mask=msk).last_hidden_state.mean(1)
        return F.normalize(x, p=2, dim=1)


model = TCEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def encode_text(text: str) -> np.ndarray:
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)
        emb = model(input_ids, attention_mask)
    return emb.cpu().numpy()[0]  # shape: (384,)

def retrieve_similar_tc(query: str, tc_data: list, top_k=3):
    query_vec = encode_text(query)
    candidates = [tc["precondition"] + " " + tc["description"] for tc in tc_data]
    cand_vecs = np.stack([encode_text(c) for c in candidates])
    
    sims = cand_vecs @ query_vec  # cosine similarity
    topk_idx = np.argsort(sims)[::-1][:top_k]

    return [tc_data[i] for i in topk_idx]
