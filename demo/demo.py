import streamlit as st
import requests

st.set_page_config(page_title="TC → Action → API Demo", layout="wide")
st.title("🧪 Test Case to API Demo")

# 입력창
pre = st.text_area("Precondition", height=100)
desc = st.text_area("Description", height=100)

if st.button("🚀 Run"):
    if not pre or not desc:
        st.warning("Precondition과 Description 모두 입력해주세요.")
    else:
        # 1. 유사 TC 검색
        with st.spinner("🔍 유사한 Test Case 찾는 중..."):
            res = requests.post("http://localhost:8000/retrieve_similar_tc", json={
                "precondition": pre,
                "description": desc
            }).json()
            examples = res["retrieved"]

        st.subheader("1️⃣ Retrieved Similar Test Cases")
        for i, ex in enumerate(examples):
            st.markdown(f"**[Example {i+1}]**")
            st.code(f"Precondition: {ex['precondition']}\nDescription: {ex['description']}\n\n→ Action Desc: {ex['action_desc']}", language="")

        # 2. Action Description 생성
        with st.spinner("🧠 Action Description 생성 중..."):
            gen = requests.post("http://localhost:8000/generate_action_desc", json={
                "current_tc": {"precondition": pre, "description": desc},
                "examples": examples
            }).json()
            action_desc = gen["generated_action_desc"]

        st.subheader("2️⃣ Generated Action Description")
        st.success(action_desc)

        # # 3. API 검색
        # with st.spinner("🔌 API 추천 중..."):
        #     api_res = requests.post("http://localhost:8000/retrieve_api", json={"action_desc": action_desc}).json()
        #     matches = api_res["matched"]

        #     st.subheader("3️⃣ Retrieved APIs")

        #     for i, match in enumerate(matches):
        #         st.markdown(f"**{i+1}. `{match['action_desc']}`**")
        #         for j, api in enumerate(match["matched_apis"]):
        #             st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- `{api['api_name']}`")


        # 3. API 검색
        with st.spinner("🔌 API 추천 중..."):
            api_res = requests.post("http://localhost:8000/retrieve_api", json={"action_desc": action_desc}).json()
            matches = api_res["sequence"]  

            st.subheader("3️⃣ Matched API Sequence")

            for i, match in enumerate(matches):
                st.markdown(f"**{i+1}. `{match['action_desc']}`** → `{match['matched_api']}`")  
