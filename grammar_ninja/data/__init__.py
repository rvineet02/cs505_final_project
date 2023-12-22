from pathlib import Path

def read_text(file_path: str):
    with open(Path(file_path), "r") as f:
        text = f.read()
    return text