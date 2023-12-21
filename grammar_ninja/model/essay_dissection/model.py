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
