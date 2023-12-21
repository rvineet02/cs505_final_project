from torch import nn
from transformers import BertModel


class BERT_Classifier(nn.Module):
    def __init__(self):
        super(BERT_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(0.0)
        self.out = nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.drop(pooled_output)
        output = self.out(output_2)
        return output
