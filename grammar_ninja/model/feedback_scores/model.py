from typing import List
import torch
from transformers import PreTrainedModel, BertConfig, BertModel, BertTokenizer
from torch import nn


class BERTClassifierConfig(BertConfig):
    model_type = "BERTClassifier"

    def __init__(self, num_labels=6, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class BERTClassifier(PreTrainedModel):
    config_class = BERTClassifierConfig
    base_model_prefix = "bert-cased"

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.drop = nn.Dropout(0.0)
        self.out = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.drop(pooled_output)
        output = self.out(output_2)
        return output

    def predict(
        self, text: str, tokenizer: BertTokenizer, device: str, max_length: int
    ) -> List[float]:
        if self.training:
            self.eval()
        with torch.no_grad():
            inputs = tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )

            ids = inputs["input_ids"]
            mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]

            ids = torch.LongTensor(ids).unsqueeze(0).to(device)
            mask = torch.LongTensor(mask).unsqueeze(0).to(device)
            token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(device)

            outputs = self.forward(ids, mask, token_type_ids)

        return outputs.cpu().detach().numpy().tolist()[0]
