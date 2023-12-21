import argparse

import gradio as gr

from tools.feedback_scores import inference


def run(input):
    if not input:
        return ("", {})
    results = inference.predict(input, model, device)
    results = inference.output(results)
    map = {
        "Cohesion": results[0] / 5,
        "Syntax": results[1] / 5,
        "Vocabulary": results[2] / 5,
        "Phraseology": results[3] / 5,
        "Grammar": results[4] / 5,
        "Conventions": results[5] / 5,
    }
    return ("", map)


examples = [
    [
        "In this passage, Aeneas visits the Underworld and has interactions with Dido and Deiphobus. Aeneas is given passage to the underworld, beyond the styx. This episode is grounded in Homer’s Odyssey where Odysseus speaks to the dead. However, in the Odyssey, Odysseus never actually goes to the underworld, he remains above land and the dead spirits appear and talk to him. First, Aeneas visits the Fields of Mourning, which is where the souls who died for love and those who committed suicide reside."
    ],
    [
        "its very important too remember that writing goodly isn't just about using big words and complex sentences, its about getting your point across in a clear and concise way. many people thinks that by using overly complicated language they can sound more smarter but this is often not the case. their are also times when writers dont use punctuation correctly, or they use run-on sentences that makes it hard for readers to follow their ideas or they switch their tense back and forth which can be confusing. Another thing is that good writing needs to have a structure like a beginning middle and end but some writers just put their thoughts down without any organization and this makes it hard to understand what they are trying to say. Also, spelling mistakes are common and it dont look professional"
    ],
]

theme = gr.themes.Soft(
    font=["Quicksand", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Hello, Welcome to Grammar Ninja!")
    gr.Markdown(
        "## Here you can improve your English Prose, by getting feedback on your writing!"
    )
    with gr.Row():
        text_input = gr.Textbox(lines=2, placeholder="Enter your text here...")
        text_output = gr.Textbox(lines=10, placeholder="Feedback will appear here...")
        output_label = gr.Label(label="Feedback Scores")
        # output_label = gr.HTML()

    gr.Button("Submit").click(
        fn=run,
        inputs=text_input,
        outputs=[text_output, output_label],
    )

    gr.Examples(examples=examples, inputs=text_input, fn=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text_path",
        type=str,
        default="test.txt",
        help="path to text file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/bert_classifier_cased.pth",
        help="path to model",
    )
    parser.add_argument(
        "--M1",
        type=int,
        default=1,
        help="flag for M1 GPU",
    )

    args = parser.parse_args()
    print("Parsing Args...")
    print(args)

    model = inference.load_model(args.model_path)
    assert model is not None, "model is None"
    device = inference.set_device(args.M1)

    demo.launch(show_api=False)
