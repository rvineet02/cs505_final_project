import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(pretrained_model_name_or_path: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        quantization_config=bnb_config,
    )
    return model


def load_tokenizer(pretrained_model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
