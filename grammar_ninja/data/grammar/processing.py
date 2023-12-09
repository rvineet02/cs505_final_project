from pathlib import Path
from typing import TypeVar, Any

PathLike = TypeVar("PathLike", str, bytes, Path)
PROMPT_TEMPLATE_PATHS = Path(__file__).parent / "prompt_templates"


class PromptTemplate:
    def __init__(self, prompt_template_path: PathLike) -> None:
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()

    @property
    def get_prompt_template(self) -> str:
        return self.prompt_template

    def format_prompt(self, **kwargs: Any) -> str:
        prompt = self.prompt_template
        for place_holder, content in kwargs.items():
            prompt = prompt.replace("{" + place_holder + "}", content)
        return prompt

    def __repr__(self) -> str:
        return self.prompt_template