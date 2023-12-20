import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from transformers import BertConfig, BertModel, BertTokenizer

from helper import getenv

APPLE_M1_FLAG = getenv("APPLE_M1_FLAG")

# load train data
file_path = "./data/feedback_scores/train.csv"
df = pd.read_csv(file_path)
print(f"Shape of the data: {df.shape}")
print()
print(f"First 5 rows of the data: \n{df.head()}")

# convert the 6 columns into 1 column with a list of 6 values
df["scores"] = df[
    ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
].values.tolist()

# remove the 6 columns
df = df.drop(
    columns=[
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
)

print(f"First 5 rows of the data after combining into a single list: \n{df.head()}")


# Dataset and Dataloader


device = None
if APPLE_M1_FLAG:
    # try to setup M1 GPU
    is_gpu = torch.backends.mps.is_available()
    if is_gpu:
        device = torch.device("mps")
        print("DEVICE: M1 GPU")
    else:
        device = torch.device("cpu")
        print("DEVICE: CPU")
else:
    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("DEVICE: GPU")
    else:
        device = torch.device("cpu")
        print("DEVICE: CPU")


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.full_text = dataframe.full_text
        self.targets = self.data.scores
        self.max_len = max_len

    def __len__(self):
        return len(self.full_text)

    def __getitem__(self, index):
        full_text = str(self.full_text[index])
        full_text = " ".join(full_text.split())

        inputs = self.tokenizer.encode_plus(
            full_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "targets": torch.LongTensor(self.targets[index]),
        }


# ------------------ HYPERPARAMETERS ------------------ #
MAX_LEN = 200
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
TRAIN_SIZE = 0.8
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


print(f"FULL Dataset: {df.shape}")

train_ds = df.sample(frac=TRAIN_SIZE, random_state=200)
test_ds = df.drop(train_ds.index).reset_index(drop=True)
train_ds = train_ds.reset_index(drop=True)

print(f"TRAIN Dataset: {train_ds.shape}")
print(f"TEST Dataset: {test_ds.shape}")

train_set = CustomDataset(train_ds, tokenizer, MAX_LEN)
test_set = CustomDataset(test_ds, tokenizer, MAX_LEN)


train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

train_loader = DataLoader(train_set, **train_params)
test_loader = DataLoader(test_set, **test_params)


# ------------------ MODEL ------------------ #
from torch import nn


class BERT_Classifier(nn.Module):
    def __init__(self):
        super(BERT_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.0)
        self.out = nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.drop(pooled_output)
        output = self.out(output_2)
        return output


model = BERT_Classifier()
model.to(device)


def loss_fn(outputs, targets):
    return nn.MSELoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# ------------------ TRAINING ------------------ #
def train(epoch):
    model.train()
    for idx, data in enumerate(train_loader, 0):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        if idx % 100 == 0:
            print(f"Batch: {idx} | Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"EPOCH: {epoch} | LOSS: {loss.item()}")


for epoch in range(EPOCHS):
    train(epoch)


# ------------------ VALIDATION ------------------ #
def validation(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            # fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def calculate_metrics(outputs, targets):
    # compute the mean squared error between the actual and predicted scores
    mse = metrics.mean_squared_error(targets, outputs)
    return mse


for epoch in range(EPOCHS):
    outputs, targets = validation(epoch=epoch)
    mse = calculate_metrics(outputs, targets)
    print(f"EPOCH: {epoch} | MSE: {mse}")
