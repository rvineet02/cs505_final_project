import transformers
from peft import (
    get_peft_model,
    LoftQConfig,
    LoraConfig,
)
from datasets import load_dataset
import os
import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
import torch
from grammar_ninja.data.grammar.preprocessing import (
    PromptTemplate,
    PROMPT_TEMPLATE_PATHS,
    generate_prompt,
)


os.environ["WANDB_PROJECT"] = "mistral-7b-grammar"

PROJECT = os.environ.get("WANDB_PROJECT")
BASE_MODEL_NAME = "mistral"
RUN_NAME = f"{BASE_MODEL_NAME}-{PROJECT}"
OUTPUT_DIR = f"../../exps/{RUN_NAME}"
LOGGING_DIR = f"../../exps/{RUN_NAME}/logs"
MAX_LENGTH = None

CACHE_DIR = "/projectnb/cs505ws/projects/grammar_ninja_alavaee/model_weights"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
PROMPT_NAME = "simple"
PROMPT_NAME = "simple"
DATA_DIR = Path(
    "/projectnb/cs505ws/projects/grammar_ninja_alavaee/data/grammar/coedit/processed"
)


if __name__ == "__main__":
    train_dataset_path = str(DATA_DIR / "train.parquet")
    val_dataset_path = str(DATA_DIR / "validation.parquet")

    dataset = load_dataset(
        "parquet",
        data_files={"train": train_dataset_path, "validation": val_dataset_path},
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
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
        "padding": True,
    }

    train_dataset = train_dataset.map(
        generate_prompt,
        fn_kwargs=fn_kwargs,
        remove_columns=["task", "instruction", "sentence", "corrected_sentence"],
    )

    val_dataset = val_dataset.map(
        generate_prompt,
        fn_kwargs=fn_kwargs,
        remove_columns=["task", "instruction", "sentence", "corrected_sentence"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        cache_dir=CACHE_DIR,
        use_cache=False,
    )

    loftq_config = LoftQConfig(loftq_bits=4)

    lora_config = LoraConfig(
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            output_dir=OUTPUT_DIR,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=500,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,  # When to start reporting loss
            logging_dir=LOGGING_DIR,  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=25,  # Save checkpoints every 25 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=25,  # Evaluate and save checkpoints every 25 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="wandb",
            run_name=f"{RUN_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    trainer.train()
