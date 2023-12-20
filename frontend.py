import argparse

import gradio as gr

from feedback_scores import inference


def run(input):
    results = inference.predict(input, model, device)
    results = inference.output(results)
    return f"Cohesion: {results[0]}\nSyntax: {results[1]}\nVocabulary: {results[2]}\nPhraseology: {results[3]}\nGrammar: {results[4]}\nConventions: {results[5]}"


markdown_text_1 = gr.Markdown("# Hello, Welcome to Grammar Ninja!")
markdown_text_2 = gr.Markdown(
    "## Here you can improve your English Prose, by getting feedback on your writing!"
)


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="Grammar Ninja",
    description="This tool helps you improve your English prose by providing feedback on various aspects like cohesion, syntax, vocabulary, etc.",
    examples=[
        [
            "In this passage, Aeneas visits the Underworld and has interactions with Dido and Deiphobus. Aeneas is given passage to the underworld, beyond the styx. This episode is grounded in Homer’s Odyssey where Odysseus speaks to the dead. However, in the Odyssey, Odysseus never actually goes to the underworld, he remains above land and the dead spirits appear and talk to him. First, Aeneas visits the Fields of Mourning, which is where the souls who died for love and those who committed suicide reside."
        ]
    ],
)
# demo = gr.Interface(fn=run, inputs="text", outputs="text")
# demo.add(markdown_text_1)
# demo.add(markdown_text_2)

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
