# Grammar Ninja

# Setup

## Environment Variables

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/home/paperspace/.cache/huggingface/transformers
```

## Download Models

```bash
huggingface-cli download lavaman131/mistral-7b-grammar
```

```bash
# install dependencies
conda env create --file config/${CONDA_ENV_FILE}
# install grammar-ninja package
pip install -e .
```