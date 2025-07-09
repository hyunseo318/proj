import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import json

# 경로 설정
DESC_ENCODER_PATH = "/work/LG/2025/output/description_encoder.pth"
API_ENCODER_PATH = "/work/LG/2025/output/api_encoder.pth"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 모델 정의 (Dual Encoder 구조 재사용)
class DualEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)

# 모델 로드
desc_encoder = DualEncoder(MODEL_NAME).to(DEVICE)
api_encoder = DualEncoder(MODEL_NAME).to(DEVICE)
desc_encoder.load_state_dict(torch.load(DESC_ENCODER_PATH, map_location=DEVICE))
api_encoder.load_state_dict(torch.load(API_ENCODER_PATH, map_location=DEVICE))
desc_encoder.eval()
api_encoder.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_desc(text: str) -> np.ndarray:
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        emb = desc_encoder(input_ids, attention_mask)
    return emb.cpu().numpy()[0]

def encode_api(text: str) -> np.ndarray:
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        emb = api_encoder(input_ids, attention_mask)
    return emb.cpu().numpy()[0]

# Load and filter API data
with open("/work/LG/2025/data/mapped_data.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

api_data = [item for item in raw_data if "action_api" in item and isinstance(item["action_api"], str) and item["action_api"].strip()]
action_api_texts = [item["action_api"] for item in api_data]
action_api_vecs = np.stack([encode_api(text) for text in action_api_texts])  # shape: (N, D)

# def retrieve_api(action_desc: str, api_data: list, top_k=3):
#     query_vec = encode_desc(action_desc)
#     api_vecs = []
#     for api in api_data:
#         api_text = api.get("action_api", "")
#         api_vecs.append(encode_api(api_text))
#     api_vecs = np.stack(api_vecs)

#     sims = api_vecs @ query_vec  # cosine similarity
#     topk_idx = np.argsort(sims)[::-1][:top_k]

#     return [api_data[i] for i in topk_idx]

def retrieve_api(action_desc: str, top_k=10):
    query_vec = encode_desc(action_desc)  # shape: (D,)
    sims = action_api_vecs @ query_vec  # shape: (N,)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [api_data[i] for i in top_idx]
