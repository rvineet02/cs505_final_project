import torch
from transformers import (
    BertModel,
    BertTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from grammar_ninja import HF_HOME
from grammar_ninja.model.feedback_scores.model import BERTClassifierConfig, BERTClassifier
from tools.grammar.inference import DEFAULT_DEVICE

MODEL_ID = "bert-base-cased"

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir=HF_HOME)
    # config = BERTClassifierConfig(num_labels=6, vocab_size=tokenizer.vocab_size)
    # model = BERTClassifier(config=config)
    # model.load_state_dict(torch.load("../../ckpts/feedback_scores/pytorch_model.bin"))
    # model.push_to_hub("lavaman131/bert-cased-writing-score")
    # tokenizer.push_to_hub("lavaman131/bert-cased-writing-score")
    model = BERTClassifier.from_pretrained("lavaman131/bert-cased-writing-score")