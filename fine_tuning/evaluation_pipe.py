import json

import pandas as pd
from answer_process import AnswerProcessor
from common import ModelHandler
from configs import DatasetConfig
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.pipelines import pipeline

from fine_tuning.data_loader import DatasetLoader


class EvaluationPipeline:
    def __init__(
        self,
        model_dir: str | None = None,
        model_id: str | None = None,
        quantization: bool = False,
        batch_size=1,
        torch_dtype="bfloat16",
        attn_implementation="eager",
    ):
        assert (
            not model_dir or not model_id
        ), "Only one of model_dir or model_id should be provided."

        self.model_source = model_dir or model_id
        self.model_handler = ModelHandler(
            model_source=self.model_source,
            quantization=quantization,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_handler.model,
            tokenizer=self.model_handler.tokenizer,
            device_map="auto",
            batch_size=batch_size,
            torch_dtype=self.model_handler.torch_dtype,
        )
        self.answer_processor = AnswerProcessor()

    def generate_answers(self, data: Dataset, cot=False) -> list[str]:
        if cot:
            raise NotImplementedError("CoT is not yet implemented.")

        answers: list[str] = []
        for example in tqdm(data, desc="Generating answers"):
            model_output = self.pipeline(
                self.model_handler.tokenizer.apply_chat_template(
                    example["text"],
                    max_new_tokens=100,
                    return_full_text=False,
                )
            )
            answers += [num["generated_text"] for num in model_output]
        return answers

    def calculate_metrics(
        self, labels: list[str], predictions: list[str]
    ) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)

    def save_results(self, result_df: pd.DataFrame, file_path: str, data_name: str, options: dict[str, bool]) -> None:
        for metric in ["f1(macro)", "acc"]:
            new_row = [
                data_name,
                metric,
                result_df.loc[result_df["data_name"] == data_name, metric].values[0],
                options["option_finetuning"],
                options["option_BitsAndBytes"],
                options["option_CoT"],
                options["option_LoRA(r=32 a=64)"],
            ]

            # Check if the row already exists in the DataFrame
            if (new_row[:2] + new_row[3:]) not in (result_df.drop("score", axis=1).values.tolist()):
                result_df.loc[-1] = new_row
                result_df.index += 1
                result_df = result_df.sort_index()
            # If it exists, update the existing row
            else:
                existed_idx = (
                    result_df.drop("score", axis=1)
                    .values.tolist()
                    .index(new_row[:2] + new_row[3:])
                )
                result_df.loc[existed_idx] = new_row

        result_df.to_csv(file_path, index=False)


if __name__ == "__main__":

    # Load models and generation configurations
    with open("fine_tuning/models.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    # Load datasets using DatasetLoader
    dataset_config = DatasetConfig()
    dataset_loader = DatasetLoader(dataset_config)

    for basemodel_id in ["google/medgemma-27b-text-it"]:#models["sota_1b_model_id_list"]:
        print(f"Evaluating {basemodel_id}...")
        
        # Load the result DataFrame
        result_df: pd.DataFrame = pd.read_csv(
            f"result_csv/{basemodel_id.split('/')[-1]}.csv"
        )

        options = {
            "option_finetuning": True,
            "option_BitsAndBytes": True,
            "option_CoT": False,
            "option_LoRA(r=32 a=64)": True,
        }

        ## float16 is recommended for AWQ
        #evaluation_pipeline = EvaluationPipeline(
        #    model_id=basemodel_id,
        #    quantization=options["option_BitsAndBytes"],
        #    batch_size=4,
        #    torch_dtype="bfloat16",
        #)

        for data_name, data in dataset_loader.test_datasets.items():
            # If the pipeline with model_id is available, disable this line
            evaluation_pipeline = EvaluationPipeline(
                model_dir=f"fine_tuned/{basemodel_id.split('/')[-1]}/{data_name}/",
                quantization=options["option_BitsAndBytes"],
                batch_size=4,
                torch_dtype="bfloat16",
                attn_implementation="eager",
            )
            
            # test 데이터의 text 컬럼 사용 (chat template이 적용된 프롬프트)
            test_texts = [example["text"] for example in data["test"]]
            answer_texts = evaluation_pipeline.generate_answers(test_texts)
            
            # label 컬럼 사용 (실제 정답)
            labels = [example["label"] for example in data["test"]]
            accuracy, f1 = evaluation_pipeline.calculate_metrics(labels, answer_texts)

            evaluation_pipeline.save_results(result_df, f"result_csv/{basemodel_id.split('/')[-1]}.csv",
                                             data_name, options)
