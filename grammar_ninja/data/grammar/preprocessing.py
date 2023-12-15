from transformers import AutoTokenizer
from pathlib import Path
from typing import TypeVar, Any, Dict, List, Union
import numpy as np
import pandas as pd
from lightning import seed_everything

seed_everything(seed=42)

PathLike = TypeVar("PathLike", str, bytes, Path)
PROMPT_TEMPLATE_PATHS = Path(__file__).parent / "prompt_templates"


class PromptTemplate:
    def __init__(self, prompt_template_path: PathLike) -> None:
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()

    @property
    def get_prompt_template(self) -> str:
        return self.prompt_template

    def format_prompt(self, placeholders: Dict[str, Any]) -> str:
        prompt = self.prompt_template
        for place_holder, content in placeholders.items():
            prompt = prompt.replace("{" + place_holder + "}", content)
        return prompt

    def __repr__(self) -> str:
        return self.prompt_template


def generate_prompt(
    example: Dict[str, Any],
    prompt_template: PromptTemplate,
    tokenizer: AutoTokenizer,
    max_length: int,
    truncation: Union[bool, str],
    padding: Union[bool, str],
) -> Dict[str, Any]:
    result = tokenizer(
        prompt_template.format_prompt(example["input_text"]),
        truncation=truncation,
        max_length=max_length,
        padding=padding,
    )
    result = dict()
    result["text"] = prompt_template.format_prompt(example)
    return result


def preprocess_coedit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["instruction"] = df["src"].str.extract(r"^(.*?):")
    df["sentence"] = df["src"].str.extract(r": (.+)")
    df.rename({"tgt": "corrected_sentence"}, axis=1, inplace=True)
    return df[["task", "instruction", "sentence", "corrected_sentence"]]


def apply_corrections(text: str, corrections: Dict[str, np.ndarray]) -> str:
    # The corrected text starts as the original text
    corrected_text = text
    offset = 0

    # Iterate over the corrections
    for start, end, correction in zip(
        corrections["start"], corrections["end"], corrections["text"]
    ):
        # Adjust start and end with the offset
        start += offset
        end += offset

        # Replace the text if correction exists
        if correction:
            corrected_text = corrected_text[:start] + correction + corrected_text[end:]
            # Update the offset
            offset += len(correction) - (end - start)

        else:
            corrected_text = corrected_text[:start] + corrected_text[end:]
            # Update the offset
            offset -= end - start

    return corrected_text


def preprocess_wi_locness_dataset(df: pd.DataFrame) -> pd.DataFrame:
    alternative_prompts = [
        "Correct the grammatical errors in the following sentence:",
        "Revise this text for proper grammar:",
        "Improve the grammatical structure of this sentence:",
        "Transform the words into grammatically correct English:",
        "Edit this sentence to eliminate grammatical mistakes:",
    ]
    corrected_text = df.apply(
        lambda row: apply_corrections(row["text"], row["edits"]), axis=1
    )
    processed_df = pd.DataFrame(
        columns=["instruction", "sentence", "corrected_sentence"]
    )
    processed_df["sentence"] = df["text"].str.replace(r"\s+", " ", regex=True)
    processed_df["corrected_sentence"] = corrected_text.str.replace(
        r"\s+", " ", regex=True
    )

    processed_df["instruction"] = np.random.choice(
        alternative_prompts, size=processed_df.shape[0], replace=True
    )

    return processed_df
