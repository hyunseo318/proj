import pandas as pd
import json

def load_tc_data(path):
    df = pd.read_excel(path)
    df = df.fillna("")
    df["precondition"] = df["precondition"].astype(str)
    df["description"] = df["description"].astype(str)
    return df.to_dict(orient="records")


def load_api_data(path="/work/LG/2025/data/mapped_data.jsonl"):
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            # 'description'은 action_description으로 대체 가능
            data.append({
                "action_api": item.get("action_api", ""),
                "description": item.get("action_description", "")  # ← optional: UI에 표시용
            })
    return data
