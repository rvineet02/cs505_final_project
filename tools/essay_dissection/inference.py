from argparse import ArgumentParser
from transformers import AutoTokenizer
from grammar_ninja import HF_HOME
from grammar_ninja.data import read_text
from grammar_ninja.model.essay_dissection.model import EssayDisectionModel
from grammar_ninja.model.utils import get_default_device

DEFAULT_DEVICE = get_default_device()
MAX_LENGTH = 205


def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    file_path = args.file_path
    text = read_text(file_path)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(
        "lavaman131/longformer-essay-dissection", cache_dir=HF_HOME
    )

    model = EssayDisectionModel.from_pretrained(
        "lavaman131/longformer-essay-dissection",
        cache_dir=HF_HOME,
        device_map=device,
    )

    results = model.predict(text=text, tokenizer=tokenizer, max_length=MAX_LENGTH) # type: ignore

    print(results)


if __name__ == "__main__":
    main()
