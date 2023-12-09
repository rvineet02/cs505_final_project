from typing import Dict
import numpy as np
import pandas as pd
from lightning import seed_everything
import re

seed_everything(seed=42)


def preprocess_coedit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["prompts"] = df["src"].str.extract(r"^(.*?):")
    df["input_text"] = df["src"].str.extract(r":(.+)")
    df.rename({"tgt": "output_text"}, axis=1, inplace=True)
    return df[["task", "prompts", "input_text", "output_text"]]


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
    processed_df = pd.DataFrame(columns=["prompts", "input_text", "output_text"])
    processed_df["input_text"] = df["text"].str.replace(r"\s+", " ")
    processed_df["output_text"] = corrected_text.str.replace(r"\s+", " ")

    processed_df["prompts"] = np.random.choice(
        alternative_prompts, size=processed_df.shape[0], replace=True
    )

    return processed_df
