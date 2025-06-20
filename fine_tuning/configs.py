from dataclasses import dataclass, field

@dataclass
class LoRAConfig:
    r: int = 4
    lora_alpha: int = 8
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    task_type: str = "CAUSAL_LM"
    
@dataclass
class DatasetConfig:
    max_seq_lengths: dict[str, int] = field(default_factory=lambda: {
        "kormedmcqa_dentist": 256,
        "kormedmcqa_doctor": 512,
        "kormedmcqa_nurse": 256,
        "kormedmcqa_pharm": 256,
        "medqa_4_options": 512,
        "medqa_5_options": 512,
        "asan_healthinfo_data": 256,
        "gen_gpt_data": 256,
        "distillation_data": 1024,
    })
    split_files: dict[str, str] = field(default_factory=lambda: {
        "train": "train.jsonl",
        "validation": "valid.jsonl",
        "test": "test.jsonl",
    })