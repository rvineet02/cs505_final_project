from collections import OrderedDict
import torch
from transformers import (
    AutoTokenizer,
)
from grammar_ninja import HF_HOME
from grammar_ninja.model.essay_dissection.model import (
    EssayDisectionConfig,
    EssayDisectionModel,
)

MODEL_ID = "bert-base-cased"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", cache_dir=HF_HOME
    )
    # config = EssayDisectionConfig(
    #     num_labels=7,
    #     hidden_dropout_prob=0.3,
    #     vocab_size=tokenizer.vocab_size,
    #     type_vocab_size=1,
    #     max_position_embeddings=4098,
    # )
    # model = EssayDisectionModel(config=config)
    # # Load the state dict
    # model.load_state_dict(torch.load("../../ckpts/essay_dissection/pytorch_model.bin"))
    # model.push_to_hub("lavaman131/longformer-essay-dissection")
    # tokenizer.push_to_hub("lavaman131/longformer-essay-dissection")
    model = EssayDisectionModel.from_pretrained(
        "lavaman131/longformer-essay-dissection", cache_dir=HF_HOME
    )
