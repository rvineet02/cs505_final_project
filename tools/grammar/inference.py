import re
import torch
import transformers
from peft import PeftModel
import spacy

from grammar_ninja.data.grammar.preprocessing import (
    PROMPT_TEMPLATE_PATHS,
    PromptTemplate,
)
from grammar_ninja.data import read_text
from argparse import ArgumentParser
from grammar_ninja.model.utils import get_default_device
from grammar_ninja import HF_HOME

MODEL_ID = "mistralai/Mistral-7B-v0.1"
FINE_TUNE_ID = "lavaman131/mistral-7b-grammar"
PROMPT_NAME = "simple"
DEFAULT_DEVICE = get_default_device()


def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    file_path = args.file_path
    text = read_text(file_path)
    device = args.device

    config = (
        transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if device == "cuda"
        else None
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        cache_dir=HF_HOME,
        quantization_config=config,
        # torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
        device_map=device,
    )

    model = PeftModel.from_pretrained(model, FINE_TUNE_ID, cache_dir=HF_HOME)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FINE_TUNE_ID,
        add_bos_token=True,
        cache_dir=HF_HOME,
        use_cache=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Replace more than one consecutive newline with a single newline
    text = re.sub(r"\n\n+", "\n", text)

    # Strip leading and trailing whitespace (including newlines)
    text = text.strip()

    paragraphs = text.split("\n")

    # tokenize the text into sentences using spacy
    nlp = spacy.load("en_core_web_md", disable=["ner", "tagger", "lemmatizer"])
    tokenized_paragraphs = []
    for doc in nlp.pipe(paragraphs, batch_size=8, n_process=2):
        tokenized_paragraphs.append([sentence.text for sentence in doc.sents])

    instructions = [
        "Remove grammar mistakes",
        "Fix coherence in this sentence",
        "Make this paragraph more neutral",
    ]

    prompt_template = PromptTemplate(
        prompt_template_path=PROMPT_TEMPLATE_PATHS / f"{PROMPT_NAME}.txt"
    )

    model.eval()

    output = []

    for paragraph in tokenized_paragraphs:
        output_paragraph = []
        for sentence in paragraph:
            for instruction in instructions:
                eval_prompt = prompt_template.format_prompt(
                    placeholders={
                        "instruction": instruction,
                        "sentence": sentence,
                        "corrected_sentence": "",
                    }
                ).strip()

                print(eval_prompt)

                model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
                input_length = model_input["input_ids"].shape[1]  # type: ignore

                # 1.5 times the input sentence number of tokens
                model_input["max_length"] = int(input_length * 3)

                with torch.no_grad():
                    sentence = tokenizer.decode(
                        model.generate(**model_input)[0][
                            input_length:
                        ],  # take only generated tokens
                        skip_special_tokens=True,
                        pad_token_id=tokenizer.pad_token,
                    )
                    print(sentence)
                raise Exception("exit")
            output_paragraph.append(sentence)
            break
        output.append(" ".join(output_paragraph))
        break
    output = "\n".join(output)

    # print(output)


if __name__ == "__main__":
    main()
