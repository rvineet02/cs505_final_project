import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import spacy

from grammar_ninja.data.grammar.preprocessing import (
    PROMPT_TEMPLATE_PATHS,
    PromptTemplate,
)

CACHE_DIR = "/home/paperspace/.cache/huggingface/transformers"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
FINE_TUNE_ID = "lavaman131/mistral-7b-grammar"
PROMPT_NAME = "simple"

if __name__ == "__main__":
    text = """NLP, it stand for Natural Language Processing, is a field in computer science, where focus on how computers can understanding and interact with human language. It's goal is to make computers can understand and respond to text or voice data. But, it's hard because languages is very complex and have many rules that often not follow logic.

In field of NLP, machine learn algorithms is used for make computers can process and analyze large amounts of natural language data. The problems is that, even with advanced algorithms, computers often don't understand the nuances, like sarcasm or idioms, in human languages. So, many times, they makes errors when they tries to interpret what a human is saying or writing."""

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        cache_dir=CACHE_DIR,
        quantization_config=config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=FINE_TUNE_ID,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, FINE_TUNE_ID, cache_dir=CACHE_DIR)

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
        "Make the sentence clear",
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

                model_input = tokenizer(eval_prompt, return_tensors="pt", pad).to("cuda")
                input_length = model_input["input_ids"].shape[1]

                # 1.5 times the input sentence number of tokens
                model_input["max_length"] = int(input_length * 3)

                with torch.no_grad():
                    sentence = tokenizer.decode(
                        model.generate(**model_input)[0][
                            input_length:
                        ],  # take only generated tokens
                        skip_special_tokens=True,
                        pad_token_id=tokenizer.pad_token
                    )
            output_paragraph.append(sentence)
        output.append(" ".join(output_paragraph))

    output = "\n".join(output)

    print(output)
