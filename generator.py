from openai import OpenAI 
import os
OPENAI_API_KEY = ''

# example: current_tc = {"precondition": ..., "description": ...}
def generate_action_description(current_tc, examples):
    prompt = """
    당신은 시스템 자동화 어시스턴트입니다.사용자가 수행한 테스트 케이스를 기반으로, 각 단계별로 수행된 행동을 구체적이고 간결하게 정리된 형태로 작성해야 합니다.\n
    각 테스트 케이스는 [Precondition] (행동 이전의 상태)과 [Description] (사용자의 실제 행동 설명)으로 구성되어 있습니다.\n
    예제를 참고하여, [Target] 케이스에 대한 Action Description을 작성해주세요.  \n\n
    """
    for ex in examples:
        prompt += f"[Example]\nPrecondition: {ex['precondition']}\nDescription: {ex['description']}\nAction Description: {ex['action_desc']}\n\n"
    
    prompt += f"[Target]\nPrecondition: {current_tc.precondition}\nDescription: {current_tc.description}\nAction Description:"

    # GPT 호출
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
