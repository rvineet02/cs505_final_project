import transformers
from peft import LoftQConfig, LoraConfig, get_peft_model
import os
import wandb

os.environ["WANDB_PROJECT"] = "mistral-7b-grammar"

project = os.environ.get("WANDB_PROJECT")
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

CACHE_DIR = "/projectnb/cs505ws/projects/grammar_ninja_alavaee/model_weights"
MODEL_ID = "mistralai/Mistral-7B-v0.1"


if __name__ == "__main__":
    tokenized_train_dataset = 
    tokenized_val_dataset = 
    
    model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    local_files_only = True,
    cache_dir=CACHE_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        local_files_only=True,
        cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    loftq_config = LoftQConfig(loftq_bits=4)
    
    lora_config = LoraConfig(
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
        )

    model = get_peft_model(model, lora_config)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=500,
            learning_rate=2.5e-5, # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25, # When to start reporting loss
            logging_dir="./logs", # Directory for storing logs
            save_strategy="steps", # Save the model checkpoint every logging step
            save_steps=25, # Save checkpoints every 25 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=25, # Evaluate and save checkpoints every 25 steps
            do_eval=True, # Perform evaluation at the end of training
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}" # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
    
    model.config.use_cache = False
    trainer.train()
