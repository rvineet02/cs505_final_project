import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

CACHE_DIR = "/home/paperspace/.cache/huggingface/transformers"
MODEL_ID = "mistralai/Mistral-7B-v0.1"

if __name__ == "__main__":
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        cache_dir=CACHE_DIR,
        quantization_config=config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
    )

    model = PeftModel.from_pretrained(
        model, "../../exps/mistral-7b-grammar/checkpoint-1000", cache_dir=CACHE_DIR
    )

    model.push_to_hub("lavaman131/mistral-7b-grammar")
    tokenizer.push_to_hub("lavaman131/mistral-7b-grammar")
