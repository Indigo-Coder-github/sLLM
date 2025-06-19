def generate_kormedmcqa_prompt(x) -> dict[str, str]:
    x["question"] = f'''### 질문: {x["question"]}
### 선택지
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}
'''
    return x
    
def generate_kormedmcqa_answer(x) -> str:
    if "explanation" in x:
        return f"### 정답: {x['answer']}: {x['answer_text']}\n### 설명: {x['explanation']}"
    else:
        return f"### 정답: {x['answer']}: {x['answer_text']}\n### 설명: "

def generate_kormedmcqa_cot_prompt(x) -> dict[str, str]:
    x["cot"] = f'''### 질문: {x["question"]}
### 선택지
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

### 정답: {x["cot"]}'''
    return x
    
def generate_medqa_5_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''### Question: {x["question"]}
### Options
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
- E: {x["options"]["E"]}
'''
    return x

def generate_medqa_4_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''### Question: {x["question"]}
### Options
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
'''
    return x

def generate_medqa_answer(x) -> str:
    if "explanation" in x:
        return f"### Answer: {x['answer_idx']}: {x['answer']}\n### Explanation: {x['explanation']}"
    else:
        return f"### Answer: {x['answer_idx']}: {x['answer']}\n### Explanation: "

def korean_chat_template(question: str, answer=None) -> list[dict[str, str]]:
    """한국어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "당신은 주어진 다지선다 문제를 풀어야 하는 학생입니다. 다음 질문을 읽고 '정답: <선택지>' 형식으로 답을 작성해주세요. 그 후에 설명을 작성해주세요.",
        },
        {"role": "user", "content": question},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})
    else:
        messages.append({"role": "assistant", "content": "### 정답: "})
    return messages

def english_chat_template(question: str, answer=None) -> list[dict[str, str]]:
    """영어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "You are a student who needs to solve the given multiple-choice question. Please read the following question and write your answer in the format 'Answer: <option>'. Then, provide an explanation.",
        },
        {"role": "user", "content": question},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})
    else:
        messages.append({"role": "assistant", "content": "### Answer: "})
    return messages

def asan_healthinfo_prompt(instruction: str, output: str) -> list[dict[str, str]]:
    """서울 아산병원 건강정보 데이터용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "당신은 의료 전문가입니다. 다음 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.",
        },
        {"role": "user", "content": f"### 질문: {instruction}"},
        {"role": "assistant", "content": f"### 답변: {output}"}
    ]
    return messages