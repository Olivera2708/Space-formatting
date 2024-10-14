import preprocessing
from sklearn.model_selection import train_test_split
from format_java import format_java_code, tokenize_and_create_input_output, tokenize_and_classify, create_vocab
import model
import torch

def create_datasets():
    data = preprocessing.process_files_in_directory("data/train")
    data['formatted'] = data['code'].apply(format_java_code)
    data = data.dropna(subset=['formatted'])
    data = data[data['formatted'] != data['code']]
    data.to_json("data/training.jsonl", orient="records", lines=True)

    data = preprocessing.load_data("data/test_orig.jsonl")
    data['formatted'] = data['code'].apply(format_java_code)
    data.to_json("test.jsonl", orient="records", lines=True)

    data = preprocessing.load_data("data/train/train0.jsonl")
    data['formatted'] = data['code'].apply(format_java_code)
    data = data.dropna(subset=['formatted'])
    data = data[data['formatted'] != data['code']]
    data.to_json("data/training.jsonl", orient="records", lines=True)

def get_input_output_training(train_data):
    vocab_stoi = {"<UNK>": 0}
    vocab_itos = {0: "<UNK>"}
    input = []
    output = []

    for d in train_data["formatted"]:
        tokens = tokenize_and_classify(d)
        vocab_stoi, vocab_itos = create_vocab(tokens, vocab_stoi, vocab_itos)
        new_input, new_output = tokenize_and_create_input_output(tokens, vocab_stoi)
        input.extend(new_input)
        output.extend(new_output)

    return input, output, vocab_stoi, vocab_itos


if __name__ == "__main__":
    # create_datasets()
    data = preprocessing.load_data("data.jsonl")

    input, output, vocab_stoi, vocab_itos = get_input_output_training(data)
    input = torch.tensor(input, dtype=torch.long)
    output = torch.tensor(output, dtype=torch.long)
    training_inputs, validation_inputs, training_outputs, validation_outputs = train_test_split(input, output, test_size=0.2, random_state=27)
