from grammar_ninja.data.grammar import preprocessing
import pandas as pd
from pathlib import Path


def main():
    DATA_DIR = Path("/projectnb/cs505ws/projects/grammar_ninja_alavaee/data/grammar")

    coedit_train = pd.read_parquet(path=DATA_DIR.joinpath("coedit", "raw", "train.parquet"))

    coedit_val = pd.read_parquet(path=DATA_DIR.joinpath("coedit", "raw", "validation.parquet"))

    wi_locness_val = pd.read_parquet(
        path=DATA_DIR.joinpath("wi_locness", "raw", "validation.parquet")
    )

    coedit_train = preprocessing.preprocess_coedit_dataset(df=coedit_train)
    save_path = DATA_DIR.joinpath("coedit", "processed", "train.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    coedit_train.to_parquet(path=save_path, index=False)

    coedit_val = preprocessing.preprocess_coedit_dataset(df=coedit_val)
    save_path = DATA_DIR.joinpath("coedit", "processed", "validation.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    coedit_val.to_parquet(path=save_path, index=False)

    wi_locness_val = preprocessing.preprocess_wi_locness_dataset(df=wi_locness_val)
    save_path = DATA_DIR.joinpath("wi_locness", "processed", "validation.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wi_locness_val.to_parquet(path=save_path, index=False)


if __name__ == "__main__":
    main()
