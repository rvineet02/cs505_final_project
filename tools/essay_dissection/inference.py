import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

LABELS = ['Claim', 'Evidence', 'Concluding Statement', 'Rebuttal', 'Position','Counterclaim', 'Lead']

model = torch.load('../data/model.pth')
model.to('mps')
model.eval()

def pad_sequence(sequence, max_len, padding_value=0):
    if len(sequence) < max_len:
        return sequence + [padding_value] * (max_len - len(sequence))
    else:
        return sequence[:max_len]


def predict(text):

    encoded = tokenizer(text, max_length=205, padding='max_length')
    padded_input = encoded['input_ids']
    mask = encoded['attention_mask']

    padded_input_tensor = torch.tensor(padded_input, dtype=torch.long).to('mps')
    mask_tensor = torch.tensor(mask, dtype=torch.long).to('mps')

    padded_input_batch = padded_input_tensor.unsqueeze(0)
    mask_batch = mask_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(padded_input_batch, mask_batch)

    predictions = torch.argmax(logits, dim=-1)

    relevant_predictions = predictions[0][mask_tensor.bool()].cpu().numpy()

    token_to_word = encoded.words()[:len(relevant_predictions)]

    word_predictions = []

    for idx, label in zip(token_to_word, relevant_predictions):
        if idx and idx >= len(word_predictions):
            word_predictions.append(label)

    return word_predictions


def visualize(text, predictions):
    words = text.split()
    tags = [labels[i] for i in predictions]

    tag_dict = {}
    current_tag = None
    sequence_number = {}

    for word, tag in zip(words, tags):
        # If the tag changes, reset the current tag and increment sequence number
        if tag != current_tag:
            current_tag = tag
            sequence_number[tag] = sequence_number.get(tag, 0) + 1
            key = f"{tag}_{sequence_number[tag]}"
            tag_dict[key] = []

        # Add the word to the current sequence
        key = f"{tag}_{sequence_number[tag]}"
        tag_dict[key].append(word)

    return tag_dict

def inference(text):
    predictions = predict(text)
    tag_dict = visualize(text, predictions)
    return tag_dict

text = "Sample text from an essay..."
prediction = predict(text)
print("Predicted label:", prediction)
visualize(text, prediction)
