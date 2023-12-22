# Grammar Ninja

# Setup

## Environment Variables

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=$HOME/.cache/huggingface/transformers
```

## Install Environment

```bash
# install dependencies
conda env create --file config/${CONDA_ENV_FILE}
# install grammar-ninja package
pip install -e .
# download spacy model
python -m spacy download en_core_web_md
```

# Demo

```bash
cd tools
```

## Essay Dissection (Longformer)

```bash
cd essay_dissection
python inference.py ../../examples/nlp.txt
```

## Feedback Scores (Bert-cased)

```bash
cd feedback_scores
python inference.py ../../examples/nlp.txt
```

## Grammar Correction (Mistral 7B)

```bash
cd grammar
python inference.py ../../examples/nlp.txt
```