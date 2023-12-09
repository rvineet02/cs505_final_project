from grammar_ninja.data.grammar import preprocessing
import pandas as pd
from pathlib import Path


def main():
    RAW_DIR = Path("/projectnb/cs505ws/projects/grammar_ninja_alavaee/data/grammar/raw")
    PROCCESS_DIR = Path(
        "/projectnb/cs505ws/projects/grammar_ninja_alavaee/data/grammar/processed"
    )

    coedit_train = pd.read_parquet(path=RAW_DIR.joinpath("coedit", "train.parquet"))

    coedit_val = pd.read_parquet(path=RAW_DIR.joinpath("coedit", "validation.parquet"))

    wi_locness_val = pd.read_parquet(
        path=RAW_DIR.joinpath("wi_locness", "validation.parquet")
    )

    coedit_train = preprocessing.preprocess_coedit_dataset(df=coedit_train)
    save_path = PROCCESS_DIR.joinpath("coedit", "train.parquet")
    if not save_path.parent.is_dir():
        save_path.parent.mkdir()
    coedit_train.to_parquet(
        path=save_path, index=False
    )

    coedit_val = preprocessing.preprocess_coedit_dataset(df=coedit_val)
    save_path = PROCCESS_DIR.joinpath("coedit", "validation.parquet")
    if not save_path.parent.is_dir():
        save_path.parent.mkdir()
    coedit_val.to_parquet(
        path=save_path, index=False
    )

    wi_locness_val = preprocessing.preprocess_wi_locness_dataset(df=wi_locness_val)
    save_path = PROCCESS_DIR.joinpath("wi_locness", "validation.parquet")
    if not save_path.parent.is_dir():
        save_path.parent.mkdir()
    wi_locness_val.to_parquet(
        path=save_path, index=False
    )


if __name__ == "__main__":
    main()
