from fastapi import FastAPI, Request
from pydantic import BaseModel
from retriever_tc import retrieve_similar_tc
from retriever_api import retrieve_api
from generator import generate_action_description
from data_loader import load_tc_data, load_api_data
import re
import numpy as np

app = FastAPI()

# Load data at startup
tc_data = load_tc_data("/work/LG/heytan_test_case_data.xlsx")
api_data = load_api_data("/work/LG/2025/data/mapped_data.jsonl")

# # 2. API 벡터 캐싱
# api_vecs = []
# filtered_api_data = []
# for api in api_data:
#     api_text = api.get("action_api", "")
#     if api_text and api_text.strip():
#         vec = encode_api(api_text)
#         api_vecs.append(vec)
#         filtered_api_data.append(api)

# api_vecs = np.stack(api_vecs)  # shape: (N, D)

class TCInput(BaseModel):
    precondition: str
    description: str

class GenInput(BaseModel):
    current_tc: TCInput
    examples: list  # list of {"precondition", "description", "action_desc"}

class ActionDescInput(BaseModel):
    action_desc: str

@app.post("/retrieve_similar_tc")
def api_retrieve_similar_tc(tc: TCInput):
    query = tc.precondition.strip() + " " + tc.description.strip()
    top_k = retrieve_similar_tc(query, tc_data)
    return {"retrieved": top_k}

@app.post("/generate_action_desc")
def api_generate_action_desc(input: GenInput):
    generated = generate_action_description(input.current_tc, input.examples)
    return {"generated_action_desc": generated}

# @app.post("/retrieve_api")
# def api_retrieve_api(input: ActionDescInput):
#     results = retrieve_api(input.action_desc, api_data) 
#     return {"retrieved_apis": results}

# @app.post("/retrieve_api")
# def api_retrieve_api(input: ActionDescInput):
#     lines = re.split(r"\n|\d+\.\s*", input.action_desc.strip())
#     lines = [line.strip() for line in lines if line.strip()]

#     matched = []
#     for line in lines:
#         apis = retrieve_api(line, api_data)
#         matched.append({
#             "action_desc": line,
#             "matched_apis": apis
#         })

#     return {"matched": matched}
@app.post("/retrieve_api")
def api_retrieve_api(input: ActionDescInput):
    lines = re.split(r"\n|\d+\.\s*", input.action_desc.strip())
    lines = [line.strip() for line in lines if line.strip()]

    matched = []
    for line in lines:
        top1_api = retrieve_api(line, api_data, top_k=10)[0]  
        matched.append({
            "action_desc": line,
            "matched_api": top1_api["action_api"] 
        })

    return {"sequence": matched}

@app.post("/retrieve_topk_api")
def api_retrieve_topk_api(input: ActionDescInput):
    lines = re.split(r"\n|\d+\.\s*", input.action_desc.strip())
    lines = [line.strip() for line in lines if line.strip()]

    matched = []
    for line in lines:
        topk_apis = retrieve_api(line, top_k=5)
        matched.append({
            "action_desc": line,
            "matched_apis": [a["action_api"] for a in topk_apis]
        })

    return {"matched": matched}

@app.post("/retrieve_top1_api")
def api_retrieve_top1_api(input: ActionDescInput):
    lines = re.split(r"\n|\d+\.\s*", input.action_desc.strip())
    lines = [line.strip() for line in lines if line.strip()]

    matched = []
    for line in lines:
        top1_api = retrieve_api(line, top_k=1)[0]
        matched.append({
            "action_desc": line,
            "matched_api": top1_api["action_api"]
        })

    return {"sequence": matched}
