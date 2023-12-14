from grammar_ninja.data.grammar.processing import PromptTemplate, PROMPT_TEMPLATE_PATHS
import pandas as pd
from pathlib import Path


def main():
    PROMPT_NAME = "simple"
    prompt_template = PromptTemplate(
        prompt_template_path=PROMPT_TEMPLATE_PATHS / f"{PROMPT_NAME}.txt"
    )

    DATA_DIR = Path(
        "/projectnb/cs505ws/projects/grammar_ninja_alavaee/data/grammar/coedit/processed"
    )

    coedit_train = pd.read_parquet(DATA_DIR.joinpath("train.parquet"))

    coedit_val = pd.read_parquet(DATA_DIR.joinpath("validation.parquet"))

    coedit_train["dataset"] = coedit_train.apply(
        lambda row: prompt_template.format_prompt(
            instruction=row["prompts"],
            sentence=row["input_text"],
            corrected_sentence=row["output_text"],
        ),
        axis=1,
    )

    save_path = DATA_DIR.joinpath(PROMPT_NAME, "train.parquet")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    coedit_train[["task", "dataset"]].to_parquet(path=save_path, index=False)

    coedit_val["dataset"] = coedit_val.apply(
        lambda row: prompt_template.format_prompt(
            instruction=row["prompts"],
            sentence=row["input_text"],
            corrected_sentence=row["output_text"],
        ),
        axis=1,
    )

    save_path = DATA_DIR.joinpath(PROMPT_NAME, "validation.parquet")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    coedit_val[["task", "dataset"]].to_parquet(path=save_path, index=False)


if __name__ == "__main__":
    main()
