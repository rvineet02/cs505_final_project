import argparse

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from grammar_ninja.model.feedback_scores.model import BERT_Classifier

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def set_device(APPLE_M1_FLAG=1):
    print("Setting device...")
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
            print("DEVICE: CUDA - GPU")
        else:
            device = torch.device("cpu")
            print("DEVICE: CPU")
    return device


def predict(text, model, device, MAX_LEN=200):
    print("Predicting...")
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
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

        outputs = model(ids, mask, token_type_ids)

        print("Done predicting...")
        return outputs.cpu().detach().numpy().tolist()[0]


def output(results):
    # every element should have 2 decimal places
    results = [round(x, 2) for x in results]
    return results


def load_model(path):
    print("Loading model...")
    model = BERT_Classifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    print("Done loading model...")
    return model


def read_text(path):
    print("Reading text from file...")
    # load contents from file into single string
    with open(path, "r") as f:
        text = f.read()
    print("Done loading text...")
    return text


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text_path",
        type=str,
        default="test.txt",
        help="path to text file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/bert_classifier_cased.pth",
        help="path to model",
    )
    parser.add_argument(
        "--M1",
        type=int,
        default=1,
        help="flag for M1 GPU",
    )

    args = parser.parse_args()
    print("Parsing Args...")
    print(args)

    text, device = None, None
    if args.text_path:
        text = read_text(args.text_path)
    if args.M1:
        device = set_device(args.M1)

    assert text is not None, "text is None"
    assert device is not None, "device is None"
    model = load_model(args.model_path)
    assert model is not None, "model is None"
    results = predict(text, model, device)
    results = output(results)
    print(results)
