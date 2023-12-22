from torch import nn
from transformers import AutoModel


class EssayDisectionModel(nn.Module):
    def __init__(self, num_labels=7, dropout_prob=0.3):
        super(EssayDisectionModel, self).__init__()

        self.backbone = AutoModel.from_pretrained(
            "allenai/longformer-base-4096",
        )

        self.dense1 = nn.Linear(self.backbone.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        backbone_output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x = self.dense1(backbone_output[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
    
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
import torch
from torch import nn

class EssayDissectionConfig(PretrainedConfig):
    model_type = "EssayDissection"

    def __init__(self, num_labels=7, hidden_dropout_prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob

class EssayDissectionModel(PreTrainedModel):
    config_class = EssayDissectionConfig
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            "allenai/longformer-base-4096",
            config=config
        )

        self.dense1 = nn.Linear(self.backbone.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = nn.Linear(256, config.num_labels)

    def forward(self, input_ids, attention_mask):
        backbone_output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x = self.dense1(backbone_output[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

