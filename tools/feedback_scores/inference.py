from typing import List
from transformers import BertConfig, BertModel, BertTokenizer
from grammar_ninja.data import read_text
from grammar_ninja.model.feedback_scores.model import BERTClassifier
from grammar_ninja.model.utils import get_default_device
from argparse import ArgumentParser
from grammar_ninja import HF_HOME

DEFAULT_DEVICE = get_default_device()
MAX_LENGTH = 200

def postprocess(results: List[float]):
    # every element should have 2 decimal places
    results = [round(x, 2) for x in results]
    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    file_path = args.file_path
    text = read_text(file_path)
    device = args.device
    
    tokenizer = BertTokenizer.from_pretrained("lavaman131/bert-cased-writing-score", cache_dir=HF_HOME)
    model = BERTClassifier.from_pretrained("lavaman131/bert-cased-writing-score",
                                           device_map=device,
                                           cache_dir=HF_HOME)

    results = model.predict( # type: ignore
        text=text, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH
    )
    results = postprocess(results=results)

    print(results)


if __name__ == "__main__":
    main()