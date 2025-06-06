import json

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from common import load_model_and_tokenizer


class FineTuner:
    def __init__(self, model_id, is_quantization=False, is_lora=False, **kwargs):
        self.model_id = model_id
        self.tokenizer = None

        if is_lora:
            self.lora_config = LoraConfig(
                r=kwargs.get("lora_r", 16),
                lora_alpha=kwargs.get("lora_alpha", 32),
                target_modules=["q_proj", "v_proj"],
                # if the r is small enough to apply to all modules, you can use the following line instead:
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
        else:
            self.lora_config = None

        self.is_quantization = is_quantization

    def load_model_and_tokenizer(self, torch_dtype="bfloat16"):
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_id=self.model_id,
            quantization=self.is_quantization,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if self.lora_config:
            self.model.add_adapter(self.lora_config)

    def _prepare_dataset(self, dataset, is_korean=True):
        def korean_chat_template(question: str) -> list[dict[str, str]]:
            korean_chat_template = [
                {
                    "role": "system",
                    "content": "다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.\n반드시 '정답: <선택지>' 형식으로 답을 먼저 작성한 후 설명을 작성하세요.",
                },
                {"role": "user", "content": f"{question}"},
            ]
            return korean_chat_template

        def english_chat_template(question: str) -> list[dict[str, str]]:
            english_chat_template = [
                {
                    "role": "system",
                    "content": "Read the following question and select the most appropriate answer from the given options.\nYou must first write the answer in the format 'Answer: <option>' and then provide an explanation.",
                },
                {"role": "user", "content": f"{question}"},
            ]
            return english_chat_template

        def apply_chat_template(example):
            if is_korean:
                messages = korean_chat_template(example["question"])
                messages.append({"role": "assistant", "content": example["answer"]})
            else:
                messages = english_chat_template(example["question"])
                messages.append({"role": "assistant", "content": example["answer"]})

            example["text"] = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return example

        return dataset.map(apply_chat_template)

    def train(
        self, train_dataset, eval_dataset, max_seq_length, is_korean=True, **kwargs
    ):
        train_dataset = self._prepare_dataset(train_dataset, is_korean=is_korean)
        eval_dataset = self._prepare_dataset(eval_dataset, is_korean=is_korean)

        sft_config = SFTConfig(
            output_dir=f"./fine_tuned/{self.model_id.split('/')[-1]}",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # deep_speed=True,  # Enable DeepSpeed if you want to use it
            torch_compile=True,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            **kwargs,
        )

        # TODO: It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
        # Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`.
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=sft_config,
            peft_config=self.lora_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()


if __name__ == "__main__":

    options = {
        "option_finetuning": True,
        "option_BitsAndBytes": True,
        "option_CoT": False,
        "option_LoRA(r=32 a=64)": False,
    }

    # Load models and their kwargs
    with open("models.json", "r") as f:
        models = json.load(f)
    with open("models_finetuning_kwargs.json", "r") as f:
        models_kwargs = json.load(f)

    split_files = {
        "train": "train.jsonl",
        "validation": "valid.jsonl",
        "test": "test.jsonl",
    }
    # Load Korean datasets
    kormedmcqa_dentist = load_dataset(
        "json", data_dir="data/KorMedMCQA/dentist/", data_files=split_files
    )
    kormedmcqa_doctor = load_dataset(
        "json", data_dir="data/KorMedMCQA/doctor/", data_files=split_files
    )
    kormedmcqa_nurse = load_dataset(
        "json", data_dir="data/KorMedMCQA/nurse/", data_files=split_files
    )
    kormedmcqa_pharm = load_dataset(
        "json", data_dir="data/KorMedMCQA/pharm/", data_files=split_files
    )

    # Load English datasets
    medqa_5_options = load_dataset(
        "json", data_dir="data/MedQA/5_options/", data_files=split_files
    )
    medqa_4_options = load_dataset(
        "json", data_dir="data/MedQA/4_options/", data_files=split_files
    )

    # max_seq_length_by_dataset
    max_seq_length_by_dataset = {
        "kormedmcqa_dentist": 256,
        "kormedmcqa_doctor": 512,
        "kormedmcqa_nurse": 256,
        "kormedmcqa_pharm": 256,
        "medqa_4_options": 512,
        "medqa_5_options": 512,
    }

    for model_id in models["sota_70b_quantized_model_id_list"]:
        fine_tuner = FineTuner(
            model_id, is_quantization=False, is_lora=False, lora_r=32, lora_alpha=64
        )
        fine_tuner.load_model_and_tokenizer("float16")

        fine_tuner.train(
            train_dataset=kormedmcqa_doctor["train"],
            eval_dataset=kormedmcqa_doctor["validation"],
            max_seq_length=max_seq_length_by_dataset["kormedmcqa_doctor"],
            **models_kwargs["sota_1b_model_kwargs"][model_id],
        )
        fine_tuner.model.save_pretrained(
            f"./fine_tuned/{model_id.split('/')[-1]}/kormedmcqa_doctor"
        )
        fine_tuner.tokenizer.save_pretrained(
            f"./fine_tuned/{model_id.split('/')[-1]}/kormedmcqa_doctor"
        )
