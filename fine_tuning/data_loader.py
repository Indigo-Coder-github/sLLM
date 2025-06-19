from h11 import Data
from configs import DatasetConfig
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets


class DatasetLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.fine_tuning_datasets = concatenate_datasets(
            [
                self.load_kormedmcqa_dataset()["dentist"]["train"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["doctor"]["train"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["nurse"]["train"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["pharm"]["train"].select_columns(["text"]),
                self.load_medqa_dataset()["medqa_5_options"]["train"].select_columns(["text"]),
                self.load_medqa_dataset()["medqa_4_options"]["train"].select_columns(["text"]),
                self.load_asan_healthinfo_dataset()["train"].select_columns(["text"]),
                self.load_gen_gpt_dataset()["train"].select_columns(["text"]),
                self.load_distillation_dataset()["train"].select_columns(["text"]),
            ]
        )
        self.validation_datasets = concatenate_datasets(
            [
                self.load_kormedmcqa_dataset()["dentist"]["validation"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["doctor"]["validation"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["nurse"]["validation"].select_columns(["text"]),
                self.load_kormedmcqa_dataset()["pharm"]["validation"].select_columns(["text"]),
                self.load_medqa_dataset()["medqa_5_options"]["validation"].select_columns(["text"]),
                self.load_medqa_dataset()["medqa_4_options"]["validation"].select_columns(["text"]),
                self.load_asan_healthinfo_dataset()["validation"].select_columns(["text"]),
                self.load_gen_gpt_dataset()["validation"].select_columns(["text"]),
            ]
        )
        self.test_datasets = DatasetDict(
            {
                "kormedmcqa_dentist": self.load_kormedmcqa_dataset()["dentist"]["test"],
                "kormedmcqa_doctor": self.load_kormedmcqa_dataset()["doctor"]["test"],
                "kormedmcqa_nurse": self.load_kormedmcqa_dataset()["nurse"]["test"],
                "kormedmcqa_pharm": self.load_kormedmcqa_dataset()["pharm"]["test"],
                "medqa_5_options": self.load_medqa_dataset()["medqa_5_options"]["test"],
                "medqa_4_options": self.load_medqa_dataset()["medqa_4_options"]["test"],
            }
        )
        
    def load_kormedmcqa_dataset(self) -> dict[str, Dataset]:
        datasets = {}
        for dataset_name in ["dentist", "doctor", "nurse", "pharm"]:
            dataset = load_dataset(
                "json",
                data_dir=f"data/KorMedMCQA/{dataset_name}/",
                data_files=self.config.split_files,
            )
            datasets[f"{dataset_name}"] = dataset

        return DatasetDict(datasets)
    
    def load_medqa_dataset(self) -> dict[str, Dataset]:
        datasets = {}
        medqa_5 = load_dataset(
            "json",
            data_dir="data/MedQA/5_options/",
            data_files=self.config.split_files,
        )
        medqa_4 = load_dataset(
            "json",
            data_dir="data/MedQA/4_options/",
            data_files=self.config.split_files,
        )
        datasets["medqa_5_options"] = medqa_5
        datasets["medqa_4_options"] = medqa_4
        
        return DatasetDict(datasets)
    
    def load_asan_healthinfo_dataset(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files="data/Asan-AMC-Healthinfo.jsonl",
            split=None
        )
        return dataset
    
    def load_gen_gpt_dataset(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files="data/GenMedGPT_5k_ko.jsonl",
        )
        return dataset
    
    def load_distillation_dataset(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_dir="data/distillation_gemini/",
            split=None
        )
        return dataset
        
if __name__ == "__main__":
    # test code
    data_loader = DatasetLoader(DatasetConfig())
    print("KorMedMCQA\n", data_loader.load_kormedmcqa_dataset())
    print("MedQA\n", data_loader.load_medqa_dataset())
    print("Asan Healthinfo\n", data_loader.load_asan_healthinfo_dataset())
    print("Gen GPT\n", data_loader.load_gen_gpt_dataset())
    print("Distillation\n", data_loader.load_distillation_dataset())
    print("Fine-tuning Datasets\n", data_loader.fine_tuning_datasets)
    print("Validation Datasets\n", data_loader.validation_datasets)
    print("Test Datasets\n", data_loader.test_datasets)