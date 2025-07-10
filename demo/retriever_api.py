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

# def retrieve_api(action_desc: str, api_data: list, top_k=3):
#     query_vec = encode_desc(action_desc)
#     api_vecs = []
#     for api in api_data:
#         # api_text = api.get("api_name", "")
#         api_text = api.get("action_api", "")  
#         if not api_text.strip():  
#             continue
#         api_vecs.append(encode_api(api_text))
#     api_vecs = np.stack(api_vecs)

#     sims = api_vecs @ query_vec  # cosine similarity
#     topk_idx = np.argsort(sims)[::-1][:top_k]

#     valid_data = [api for api in api_data if api.get("action_api", "").strip()]
#     return [valid_data[i] for i in topk_idx]
#     # return [api_data[i] for i in topk_idx]


# ✅ API 데이터 로드 + 유니크 필터링 + 사전 임베딩
with open("/work/LG/2025/data/mapped_data.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

api_data = []
action_api_texts = []
seen = set()

for item in raw_data:
    api = item.get("action_api")
    if not isinstance(api, str):
        continue  # None, 숫자, 리스트 등은 제외
    api = api.strip()
    if not api or api in seen:
        continue  # 빈 문자열 or 중복 제거
    seen.add(api)
    api_data.append({
        "action_api": api,
        "description": item.get("action_description", "")
    })
    action_api_texts.append(api)

# ✅ 사전 임베딩된 API 벡터
action_api_vecs = np.stack([encode_api(text) for text in action_api_texts])  # shape: (N, D)

# ✅ 검색 함수
def retrieve_api(action_desc: str, top_k=10):
    query_vec = encode_desc(action_desc)  # shape: (D,)
    sims = action_api_vecs @ query_vec    # cosine similarity
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [api_data[i] for i in top_idx]
