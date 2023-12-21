import re
import torch
import ctransformers
import transformers
from peft import PeftModel
import spacy

from grammar_ninja.data.grammar.preprocessing import (
    PROMPT_TEMPLATE_PATHS,
    PromptTemplate,
)
from argparse import ArgumentParser

CACHE_DIR = "/Users/alilavaee/.cache/huggingface/transformers"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
FINE_TUNE_ID = "lavaman131/mistral-7b-grammar"
PROMPT_NAME = "simple"

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


def main():
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    text = args.text
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
        cache_dir=CACHE_DIR,
        quantization_config=config,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # model = ctransformers.AutoModelForCausalLM.from_pretrained(
    #     "TheBloke/Mistral-7B-v0.1-GGUF",
    #     model_file="mistral-7b-v0.1.Q4_K_M.gguf",
    #     model_type="mistral",
    #     gpu_layers=50,
    # )

    model = PeftModel.from_pretrained(model, FINE_TUNE_ID, cache_dir=CACHE_DIR)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FINE_TUNE_ID,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
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
        # "Fix coherence in this sentence",
        # "Make the sentence clear",
        # "Make this paragraph more neutral",
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

                model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
                input_length = model_input["input_ids"].shape[1]  # type: ignore

                # 1.5 times the input sentence number of tokens
                model_input["max_length"] = int(input_length * 3)

                with torch.no_grad():
                    # generation = model.generate(model_input["input_ids"][0])
                    # generation = [g for g in generation]
                    # print(tokenizer.decode(generation))
                    sentence = tokenizer.decode(
                        model.generate(**model_input, repetition_penalty=2.0)[0][
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
