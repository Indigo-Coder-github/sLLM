import json

from common import ModelHandler
from configs import DatasetConfig, LoRAConfig
from data_loader import DatasetLoader
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


class FineTuner:
    def __init__(self, model_id, is_quantization=False, is_lora=False, **kwargs):
        self.model_id = model_id
        self.model_handler = None

        if is_lora:
            lora_config = LoRAConfig(
                r=kwargs.get("lora_r", 32),
                lora_alpha=kwargs.get("lora_alpha", 64),
            )
            self.lora_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                task_type=lora_config.task_type,
            )
        else:
            self.lora_config = None

        self.is_quantization = is_quantization
    
    def load_model_and_tokenizer(self, torch_dtype="bfloat16"):
        self.model_handler = ModelHandler(
            model_source=self.model_id,
            quantization=self.is_quantization,
            torch_dtype=torch_dtype,
        )

    def apply_chat_template(self, x):
        x["text"] = self.model_handler.tokenizer.apply_chat_template(x["text"], tokenize=False, return_tensors="pt")
        
        return x

    def train(self, train_dataset, eval_dataset, max_seq_length, **kwargs):
        train_dataset = train_dataset.map(self.apply_chat_template)
        eval_dataset = eval_dataset.map(self.apply_chat_template)

        sft_config = SFTConfig(
            output_dir=f"./fine_tuned/{self.model_id.split('/')[-1]}",
            overwrite_output_dir=True,
            save_total_limit=1,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            torch_compile=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            report_to="none",
            bf16=True,
            **kwargs,
        )
        
        trainer = SFTTrainer(
            model=self.model_handler.model,
            processing_class=self.model_handler.tokenizer,
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
        "option_LoRA(r=32 a=64)": True,
    }

    # Load models and their kwargs
    with open("models.json", "r") as f:
        models = json.load(f)
    with open("models_finetuning_kwargs.json", "r") as f:
        models_kwargs = json.load(f)

    # Load datasets using DatasetLoader
    dataset_config = DatasetConfig()
    dataset_loader = DatasetLoader(dataset_config)
    
    # Load lora configuration
    lora_config = LoRAConfig()
    
    for model_id in ["google/medgemma-4b-it"]:#models["sota_70b_quantized_model_id_list"]:
        fine_tuner = FineTuner(
            model_id,
            is_quantization=options["option_BitsAndBytes"],
            is_lora=options["option_LoRA(r=32 a=64)"],
            lora_r=lora_config.r,
            lora_alpha=lora_config.lora_alpha
        )
        fine_tuner.load_model_and_tokenizer()

        fine_tuner.train(
            train_dataset=dataset_loader.fine_tuning_datasets,
            eval_dataset=dataset_loader.validation_datasets,
            max_seq_length=1536,
            **models_kwargs["medical_model_kwargs"][model_id],
        )