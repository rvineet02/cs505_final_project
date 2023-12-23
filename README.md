# Grammar Ninja

<div align="center">
<img src="images/grammar_ninja.png" width="400" height="400">
</div>

# Introduction

Improving one’s English language writing is a significant challenge without access to proficient teachers that can provide valuable feedback. Given the recent rapid acceleration in generative models ability to understand language, we aim to develop a model/fine-tune a model to provide an interface that will generate feedback given text as an input. Our goal is provide quantitative benchmarks for language proficiency in six different areas: cohesion, syntax, vocabulary, phraseology, grammar, and conventions in a provided writing. Additionally, we also generate feedback at the inference layers to provide concrete feedback as to how the input text can be improved. To conclude, our project aims to apply the concepts learned in class to a real-world challenge, by providing a interface to acquire feedback on English writing. By focusing on key areas of language skills and providing model generated actionable feedback, we hope to contribute a somewhat practical tool. 

This project aims to develop a Natural Language Processing (NLP) based system that can automatically evaluate and provide feedback on student argumentative essays. The system will focus on several key aspects of writing, including the effectiveness of arguments, grammar, use of evidence, syntax, and tone. The feedback can be either quantitative, in the form of scores in various categories, or qualitative, as generated English feedback that offers specific guidance and suggestions for improvement.

For more specific and technical details about the project refer to [the report here](final_report.md).

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