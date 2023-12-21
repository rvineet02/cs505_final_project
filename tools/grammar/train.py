import transformers
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoftQConfig,
    LoraConfig,
)
from datasets import load_dataset
import os
import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
from datetime import datetime
import torch
from grammar_ninja.data.grammar.preprocessing import (
    PromptTemplate,
    PROMPT_TEMPLATE_PATHS,
    generate_prompt,
)

os.environ["WANDB_PROJECT"] = "mistral-7b-grammar"

RUN_NAME = "mistral-7b-grammar-alpaca"
OUTPUT_DIR = f"../../exps/{RUN_NAME}"
LOGGING_DIR = f"../../exps/{RUN_NAME}/logs"
MAX_LENGTH = 200

HF_HOME = os.environ.get("HF_HOME", None)
assert HF_HOME is not None, "HF_HOME is not set"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
PROMPT_NAME = "simple"
DATA_DIR = Path("/data/grammar/coedit/processed")
TASKS = ["gec", "neutralize", "coherence"]


if __name__ == "__main__":
    train_dataset_path = str(DATA_DIR / "train.parquet")
    val_dataset_path = str(DATA_DIR / "validation.parquet")

    dataset = load_dataset(
        "parquet",
        data_files={"train": train_dataset_path, "validation": val_dataset_path},
    )

    dataset = dataset.filter(lambda row: row["task"] in TASKS)

    train_dataset = dataset["train"]  # type: ignore
    val_dataset = dataset["validation"]  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        padding_side="left",
        add_bos_token=True,
        add_eos_token=True,
        cache_dir=HF_HOME,
        use_cache=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    prompt_template = PromptTemplate(
        prompt_template_path=PROMPT_TEMPLATE_PATHS / f"{PROMPT_NAME}.txt"
    )

    fn_kwargs = {
        "prompt_template": prompt_template,
        "tokenizer": tokenizer,
        "max_length": MAX_LENGTH,
        "truncation": True,
        "padding": "max_length",
    }

    train_dataset = train_dataset.map(generate_prompt, fn_kwargs=fn_kwargs)  # type: ignore

    val_dataset = val_dataset.map(generate_prompt, fn_kwargs=fn_kwargs)  # type: ignore

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        cache_dir=HF_HOME,
        use_cache=False,
        quantization_config=config,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=val_dataset,  # type: ignore
        args=transformers.TrainingArguments(
            output_dir=OUTPUT_DIR,
            warmup_steps=1,
            per_device_train_batch_size=8,  # Batch size,
            gradient_accumulation_steps=1,  # No gradient accumulation steps,
            gradient_checkpointing=True,
            # max_steps=1000,  # Total number of training steps
            num_train_epochs=1,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=True,  # Use mixed precision training with bfloat16,
            optim="paged_adamw_8bit",  # Use 8-bit AdamW
            logging_steps=250,  # When to start reporting loss
            logging_dir=LOGGING_DIR,  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=1000,  # Save checkpoints
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=1000,  # Evaluate and save checkpoints
            do_eval=True,  # Perform evaluation at the end of training
            report_to="wandb",  # type: ignore
            run_name=f"{RUN_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    trainer.train()
