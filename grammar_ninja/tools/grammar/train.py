from grammar_ninja.data.grammar.processing import PromptTemplate, PROMPT_TEMPLATE_PATHS
from grammar_ninja.model.grammar.mistral_7b import load_model, load_tokenizer


if __name__ == "__main__":
    model = load_model(pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1")
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1"
    )

    prompt_template = PromptTemplate(
        prompt_template_path=PROMPT_TEMPLATE_PATHS / "template.txt"
    )

    prompt_template.format_prompt(
        sentence="Hellow there!", corrected_sentence="Hello there!"
    )

    print()
    print(prompt_template)
