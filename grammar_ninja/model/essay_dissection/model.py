from typing import Dict, List
import torch
from transformers import (
    PreTrainedModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
)
from torch import nn

LABELS = [
    "Claim",
    "Evidence",
    "Concluding Statement",
    "Rebuttal",
    "Position",
    "Counterclaim",
    "Lead",
]


class EssayDisectionConfig(LongformerConfig):
    model_type = "EssayDisectionModel"

    def __init__(self, num_labels=7, hidden_dropout_prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob


class EssayDisectionModel(PreTrainedModel):
    config_class = EssayDisectionConfig
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)
        self.backbone = LongformerModel.from_pretrained(
            "allenai/longformer-base-4096", config=config
        )
        self.dense1 = nn.Linear(self.backbone.config.hidden_size, 256)  # type: ignore
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = nn.Linear(256, config.num_labels)

    def forward(self, input_ids, attention_mask):
        backbone_output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )  # type: ignore
        x = self.dense1(backbone_output[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

    def _predict_words(
        self, text: str, tokenizer: LongformerTokenizer, max_length: int
    ) -> List[str]:
        if self.training:
            self.eval()

        encoded = tokenizer(text, max_length=max_length, padding="max_length")
        padded_input = encoded["input_ids"]
        mask = encoded["attention_mask"]

        padded_input_tensor = torch.tensor(padded_input, dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.long).to(self.device)

        padded_input_batch = padded_input_tensor.unsqueeze(0)
        mask_batch = mask_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(padded_input_batch, mask_batch)

        predictions = torch.argmax(logits, dim=-1)

        relevant_predictions = predictions[0][mask_tensor.bool()].cpu().numpy()

        token_to_word = encoded.words()[: len(relevant_predictions)]

        word_predictions = []

        for idx, label in zip(token_to_word, relevant_predictions):
            if idx and idx >= len(word_predictions):
                word_predictions.append(label)

        return word_predictions

    def _visualize(self, text: str, predictions: List[str]):
        words = text.split()
        tags = [LABELS[i] for i in predictions]  # type: ignore

        tag_dict = {}
        current_tag = None
        sequence_number = {}

        for word, tag in zip(words, tags):
            # If the tag changes, reset the current tag and increment sequence number
            if tag != current_tag:
                current_tag = tag
                sequence_number[tag] = sequence_number.get(tag, 0) + 1
                key = f"{tag}_{sequence_number[tag]}"
                tag_dict[key] = []

            # Add the word to the current sequence
            key = f"{tag}_{sequence_number[tag]}"
            tag_dict[key].append(word)

        return tag_dict

    def predict(
        self, text: str, tokenizer: LongformerTokenizer, max_length: int
    ) -> Dict[str, List[str]]:
        predictions = self._predict_words(text, tokenizer, max_length)
        tag_dict = self._visualize(text, predictions)
        return tag_dict
