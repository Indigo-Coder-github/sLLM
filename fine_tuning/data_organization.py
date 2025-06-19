import json
import os
from pathlib import Path

import pandas as pd
from chat_templates import *

# Constants
RAW_DATA_DIR = "raw_data"
OUTPUT_DATA_DIR = "data"
KORMEDMCQA = "KorMedMCQA"
MEDQA = "MedQA"

def ensure_dir_exists(path: str):
    """디렉토리가 존재하지 않으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)

def write_jsonl(data: pd.DataFrame, filepath: str, columns: list):
    """DataFrame을 JSONL 형식으로 저장"""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data[columns].itertuples(index=False):
            row_dict = {columns[i]: getattr(row, columns[i]) for i in range(len(columns))}
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

def write_jsonl_with_templates(data: pd.DataFrame, filepath: str, columns: list, is_korean: bool = True):
    """DataFrame을 chat template이 적용된 JSONL 형식으로 저장"""
    ensure_dir_exists(os.path.dirname(filepath)) 
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data[columns].itertuples(index=False):
            question = getattr(row, columns[0])
            answer = getattr(row, columns[1]) if len(columns) > 1 else None
            # 한국어와 영어에 따라 다른 템플릿 사용
            if is_korean:
                messages = korean_chat_template(question, answer)
            else:
                messages = english_chat_template(question, answer)
            row_dict = {"text": messages, "label": answer}

            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

def process_kormedmcqa_data(dataset_folder: str, data: str, file_type: str):
    """KorMedMCQA 데이터 처리"""
    input_file = f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{file_type}.csv"
    output_file = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{file_type}.jsonl"
    
    df = pd.read_csv(input_file, index_col=0)
    
    # label은 선지 한 글자만 (A, B, C, D, E)
    df["label"] = df["answer"]
    
    # train/valid는 정답 포함, test는 정답 제외
    if file_type in ["train", "valid"]:
        df["answer_text"] = [df.loc[row_idx, df.iloc[row_idx]["answer"]] for row_idx in range(len(df))]
        df["answer"] = df.apply(generate_kormedmcqa_answer, axis=1)
        df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
        df["text"] = df.apply(lambda x: korean_chat_template(x["question"], x["answer"]), axis=1)
    else:  # test
        df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
        df["text"] = df.apply(lambda x: korean_chat_template(x["question"], None), axis=1)
    
    write_jsonl(df, output_file, ["text", "label"])

def process_medqa_data(dataset_folder: str, data: str, input_file: str, output_file: str):
    """MedQA 데이터 처리"""
    with open(f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{input_file}", "r", encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    
    # label은 선지 한 글자만 (A, B, C, D, E)
    df["label"] = df["answer_idx"]
    
    if data == "4_options":
        df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
    elif data == "5_options":
        df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
    
    # train/valid는 정답 포함, test는 정답 제외
    if "train" in input_file or "dev" in input_file:
        df["answer"] = df.apply(generate_medqa_answer, axis=1)
        df["text"] = df.apply(lambda x: english_chat_template(x["question"], x["answer"]), axis=1)
    else:  # test
        df["text"] = df.apply(lambda x: english_chat_template(x["question"], None), axis=1)
    
    output_path = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{output_file}"
    write_jsonl(df, output_path, ["text", "label"])

def process_datasets(kormedmcqa_config: dict, medqa_config: dict):
    """데이터셋을 처리하는 공통 함수"""
    for dataset_folder in os.listdir(RAW_DATA_DIR):
        if dataset_folder == KORMEDMCQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_kormedmcqa_data(dataset_folder, data, **kormedmcqa_config)
        elif dataset_folder == MEDQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_medqa_data(dataset_folder, data, **medqa_config)

def organize_data_by_type(kormedmcqa_file_type: str, medqa_input_file: str, medqa_output_file: str):
    """데이터 타입별로 데이터를 조직화하는 범용 함수"""
    kormedmcqa_config = {"file_type": kormedmcqa_file_type}
    medqa_config = {"input_file": medqa_input_file, "output_file": medqa_output_file}
    process_datasets(kormedmcqa_config, medqa_config)

def organize_cot_data():
    for dataset_folder in os.listdir(f"{RAW_DATA_DIR}/{KORMEDMCQA}"):
        input_file = f"{RAW_DATA_DIR}/{KORMEDMCQA}/{dataset_folder}/fewshot.csv"
        output_file = f"{OUTPUT_DATA_DIR}/{KORMEDMCQA}/{dataset_folder}/fewshot.jsonl"
        
        df = pd.read_csv(input_file, index_col=0)
        df["cot"] = df.apply(generate_kormedmcqa_cot_prompt, axis=1)["cot"]
        
        write_jsonl(df, output_file, ["cot"])

def organize_asan_healthinfo_data():
    input_file = f"{RAW_DATA_DIR}/Asan-AMC-Healthinfo.csv"
    output_file = f"{OUTPUT_DATA_DIR}/Asan-AMC-Healthinfo.jsonl"

    df = pd.read_csv(input_file)
    df["text"] = df.apply(lambda x: asan_healthinfo_prompt(x["instruction"], x["output"]), axis=1)
    df["label"] = df["output"]
    write_jsonl(df, output_file, ["text", "label"])
    
def organize_gengpt_data():
    input_file = f"{RAW_DATA_DIR}/GenMedGPT-5k-ko.csv"
    output_file = f"{OUTPUT_DATA_DIR}/GenMedGPT_5k_ko.jsonl"

    df = pd.read_csv(input_file)
    df["text"] = df.apply(lambda x: asan_healthinfo_prompt(x["input"], x["output"]), axis=1)
    df["label"] = df["output"]

    write_jsonl(df, output_file, ["text", "label"])
    
def organize_distillation_data():
    for dataset_file in os.listdir(f"{RAW_DATA_DIR}/distillation_gemini"):
        if "medqa" in dataset_file: continue #MedQA 데이터 제외
        
        input_file = f"{RAW_DATA_DIR}/distillation_gemini/{dataset_file}"
        output_file = f"{OUTPUT_DATA_DIR}/distillation_gemini/{dataset_file}"
        
        df = pd.read_json(input_file, lines=True)
        df["text"] = df.apply(lambda x: korean_chat_template(x["question"], x["answer"]), axis=1)

        write_jsonl(df, output_file, ["text", "label"])
    
if __name__ == "__main__":
    organize_data_by_type("train", "train.jsonl", "train.jsonl")
    organize_data_by_type("valid", "dev.jsonl", "valid.jsonl")
    organize_data_by_type("test", "test.jsonl", "test.jsonl")
    organize_cot_data()
    organize_asan_healthinfo_data()
    organize_gengpt_data()
    organize_distillation_data()