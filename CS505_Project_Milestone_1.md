# CS505 Project Milestone 1

Class: CS505
Materials: https://www.cs.bu.edu/fac/snyder/cs505/final_project.html
Reviewed: Yes

# Names:

- Alex Lavaee
- Vineet Raju
- Dhruv Chandwani

# Brief Description:

Improving one’s English language writing is a significant challenge without access to proficient teachers that can provide valuable feedback. Given the recent rapid acceleration in generative models ability to understand language, we aim to develop a model/fine-tune a model to provide an interface that will generate feedback given text as an input. Our goal is provide quantitative benchmarks for language proficiency in six different areas: cohesion, syntax, vocabulary, phraseology, grammar, and conventions in a provided writing. Additionally, we also generate feedback at the inference layers to provide concrete feedback as to how the input text can be improved. To conclude, our project aims to apply the concepts learned in class to a real-world challenge, by providing a interface to acquire feedback on English writing. By focusing on key areas of language skills and providing model generated actionable feedback, we hope to contribute a somewhat practical tool.

# Resources:

## Research Papers

### Longformer

[](https://arxiv.org/pdf/2004.05150.pdf)

### Llama 2

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

### T5

[](https://arxiv.org/pdf/1910.10683.pdf)

### Essay feedback

[](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2731315.pdf)

## Blogs/Tutorials

### Longformer

[ELL-NLP (multioutput)](https://www.kaggle.com/code/tracyporter/ell-nlp-multioutput)

[TensorFlow - LongFormer - NER - [CV 0.633]](https://www.kaggle.com/code/cdeotte/tensorflow-longformer-ner-cv-0-633)

[two longformers are better than 1](https://www.kaggle.com/code/abhishek/two-longformers-are-better-than-1)

### Llama 2

[Getting started with Llama 2 - AI at Meta](https://ai.meta.com/llama/get-started/)

### Model training

[Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)

[](https://github.com/huggingface/transformers/tree/main/examples/pytorch)

[Load adapters with 🤗 PEFT](https://huggingface.co/docs/transformers/peft)

# Project Plan:

### Problem Statement:

This project aims to develop a Natural Language Processing (NLP) based system that can automatically evaluate and provide feedback on student argumentative essays. The system will focus on several key aspects of writing, including the effectiveness of arguments, grammar, use of evidence, syntax, and tone. The feedback can be either quantitative, in the form of scores in various categories, or qualitative, as generated English feedback that offers specific guidance and suggestions for improvement.

### Datasets:

Score writing:

[Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)

Identifying parts of writing (argument, evidence, rebuttal, etc):

[Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness)

[Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021)

Grammar, style, and writing improvement datasets:

[grammarly/coedit · Datasets at Hugging Face](https://huggingface.co/datasets/grammarly/coedit)

[wi_locness · Datasets at Hugging Face](https://huggingface.co/datasets/wi_locness)

[leslyarun/c4_200m_gec_train100k_test25k · Datasets at Hugging Face](https://huggingface.co/datasets/leslyarun/c4_200m_gec_train100k_test25k)

### Data Wrangling:

- Datasets:
  - [https://www.kaggle.com/competitions/feedback-prize-effectiveness/data?select=train.csv](https://www.kaggle.com/competitions/feedback-prize-effectiveness/data?select=train.csv)
    ![Screenshot 2023-11-30 at 11.09.28 AM.png](./images/CS505%20Project%20Milestone%201%20020eb00dd91946f0b52eb2650c44c26c/Screenshot_2023-11-30_at_11.09.28_AM.png)
  - [https://www.kaggle.com/competitions/feedback-prize-2021/data?select=train.csv](https://www.kaggle.com/competitions/feedback-prize-2021/data?select=train.csv)
    ![Screenshot 2023-11-30 at 11.11.38 AM.png](./images/CS505%20Project%20Milestone%201%20020eb00dd91946f0b52eb2650c44c26c/Screenshot_2023-11-30_at_11.11.38_AM.png)

The data sources in focus right now have different labels and methods for classifying different parts of the arguments. All the data will have to be pre-processed to follow the same standard for dissection

- Models - our project will potentially have three individual models:
  - Given a piece of text identify which part of the argument it is
    - Input - A part of an essay
    - Output - Identification of type of the part - claim, thesis, body, lead, etc
  - Given a text give back argument scores
    - Input - An essay along with some pointers that dissect the parts of the essay for the model
    - Output - Scores for different aspects of the argument such as cohesion, syntax, vocabulary, phraseology, grammar, etc
  - Given a text give back written feedback
    - Input - An essay along with some pointers that dissect the parts of the essay for the model (also potentially the scores from the previous model)
    - Output - written feedback to improve the effectiveness of the argument

### Methods/Algorithms:

- Learning method (fine-tuning)
  - Generation (unsupervised learning)
    - Input: sequence
    - Output: generation sequence
  - Generation (supervised learning)
    - Inspired by T5 architecture (Text-to-Text Transformer Architecture)
      - Represent learning problem as generating text given:
        - Input - “Task description: Task input”
        - Output - “Expected output”
  - Multi-output model
    - classification of text
    - identify start and stop of sequence
    - BERT variants
- Two-phase architecture for evaluating writing and giving feedback
  - Part 1: Score different aspects of current writing
  - Part 2: Correct the writing for cohesion, style, grammar, etc.

### Training/Inference for Model:

How are we going to train the models?

- HuggingFace library with parameter efficient tuning methods (PEFT)

As we mention above, we will use

- [BERT](https://huggingface.co/bert-base-uncased) - use it to score the text using writing score dataset
  - fine-tune regression on input text for scoring different aspects of current writing
- [Llama 2](https://huggingface.co/docs/transformers/model_doc/llama2), [T5](https://huggingface.co/grammarly/coedit-xxl), [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) - use it to improve the cohesion, grammar, and style of writing
  - fine-tune generation on datasets for cohesion, style, grammar, etc.

What compute?

- SCC

What about inference?

- serverless functions for GPU offered by Cloudflare potentially
- if first option not possible, use CPU serverless functions

Create simple website for app demo

- rough structure of website
  - text box or file upload
  - UI for text output and scores

### Evaluation Strategy:

As discussed above, we aim to use the final output as a mixture of 3 different models, so we need to be able to identify and evaluate individual aspects of the model.

Model for Identifying Argumentative Aspects of Input Text:

- We can use statistics like precision, recall and f1-scores to access the ability of our model to segment text into discrete elements. These metrics can also be used for classification accuracy. Then, we can use cross-validation to compute the degree to which the model is over-fitting. We can also use error analysis techniques to identify the types of mistakes the model is making to identify areas for improvement.

Model for Scoring Argumentative Text:

- After identifying the portion of the text which is argumentative, we can use regression analysis (Mean Squared Error, Root Mean Squared Error, Mean Absolute Error) since we are predicting regression scores.
- We can also correlate our model with human scores using Pearson’s correlation coefficient or even Spearman’s coefficient so as to evaluate the model’s alignment with human judgement.
- We can also feature analysis to identify which features of text are more predictive of higher scores to gain insights into how the model is making decisions.

Model for Generating Written Feedback:

- We can the use BiLingual Evaluation Understudy (BLEU) scores to evaluate the generated feedback. It is technique to evaluate machine-translated/generated text. It provides a score between 0 and 1 to measure the similarity of the generated text to the reference feedback.
- We can also use Edit distance between the model’s output and the target text to identify how many changes the model needs to make to match the target.
- We can also use human feedback to do some reinforcement by human feedback to help the model understand the format of how feedback should be like.

As discussed above, we can use the human feedback more broadly as well for all models, thus will help with alignment as well. Then, generally, we also plan to employ higher-level evaluation strategies as well - model comparison between different intermediate techniques (various NN architectures), robustness testing with out-of-sample data, and runtime and resource usage as well. This will improve the generalizability for the models. Additionally, another strategy we can employ to evaluate our model is to compare it to chatGPT 4, so we can compare performance from a general LLM to our more specialized model.

### Github:

[](https://github.com/rvineet02/cs505_final_project)

# Tasks Split:

> From the requirements: Be sure to state, if there is more than one team member, how you will divide up the work.

- [x] @Vineet Raju @Dhruv Chandwani @Alex Lavaee Data Pre-processing
- [x] @Alex Lavaee Setting up SCC compute
- [x] @Alex Lavaee Setting up Python environment
- [x] @Vineet Raju Setup BERT with score [Feedback](https://www.notion.so/CS505-Project-Milestone-1-020eb00dd91946f0b52eb2650c44c26c?pvs=21) dataset
- [x] @Dhruv Chandwani Setup Longformer with [Sections of Writing](https://www.notion.so/CS505-Project-Milestone-1-020eb00dd91946f0b52eb2650c44c26c?pvs=21) dataset
  - [x] Remember to give feedback based on section of writing that is missing (i.e. evidence is missing) by feeding it to Llama?
- [x] @Alex Lavaee Setup Mistral 7B with [Grammarly](https://www.notion.so/CS505-Project-Milestone-1-020eb00dd91946f0b52eb2650c44c26c?pvs=21) dataset
- [x] @Vineet Raju @Dhruv Chandwani @Alex Lavaee Prepare Frontend
