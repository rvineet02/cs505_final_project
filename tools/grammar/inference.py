import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

CACHE_DIR = "/home/paperspace/.cache/huggingface/transformers"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
FINE_TUNE_ID = "lavaman131/mistral-7b-grammar"

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
        pretrained_model_name_or_path=FINE_TUNE_ID,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
    )

    model = PeftModel.from_pretrained(model, FINE_TUNE_ID, cache_dir=CACHE_DIR)

    eval_prompt = """Remove all grammatical errors from this text

### Sentence:
Hellow there me name is Alex

### Corrected Sentence:
"""

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=32)[0],
                skip_special_tokens=True,
            )
        )
