import preprocessing
from sklearn.model_selection import train_test_split
from format_java import format_java_code, tokenize_sentence_and_create_input_output, tokenize_and_classify, create_vocab
from model import TokenSpacingModel
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence

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
    vocab_stoi = {"<PAD>": 0, "<UNK>": 1}
    vocab_itos = {0: "<PAD>", 1: "<UNK>"}
    input = []
    output = []

    for d in train_data["formatted"]:
        tokens = tokenize_and_classify(d)
        vocab_stoi, vocab_itos = create_vocab(tokens, vocab_stoi, vocab_itos)
        new_input, new_output = tokenize_sentence_and_create_input_output(tokens, vocab_stoi)

        input.extend(new_input)
        output.extend(new_output)

    return input, output, vocab_stoi, vocab_itos

def train(model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, optimizer, device):
    model.to(device)
    
    dataset = TensorDataset(training_inputs, training_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_batch, output_batch in dataloader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            optimizer.zero_grad()
            spacing_type_pred = model(input_batch)

            predictions_reshaped = spacing_type_pred.reshape(-1, 4)
            correct_reshaped = output_batch.reshape(-1)

            loss_type = loss_fn_type(predictions_reshaped, correct_reshaped)
            total_loss += loss_type.item()

            loss_type.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def evaluate(model, eval_inputs, eval_outputs, batch_size, device, length_threshold=0.1):
    model.eval()
    dataset = TensorDataset(eval_inputs, eval_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    correct_type_top1 = 0
    correct_type_top3 = 0
    total_type = 0
    
    with torch.no_grad():
        for input_batch, output_batch in dataloader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            spacing_type_pred = model(input_batch)
            predictions_reshaped = spacing_type_pred.reshape(-1, 4)

            _, top_k_indices = torch.topk(predictions_reshaped, k=3, dim=1, largest=True, sorted=True)
            correct_reshaped = output_batch.reshape(-1)
            total_type += correct_reshaped.size(0)

            correct_type_top1 += (top_k_indices[:, 0] == correct_reshaped).sum().item()
            correct_type_top3 += (top_k_indices == correct_reshaped.unsqueeze(1)).sum().item()

    top1_accuracy_type = correct_type_top1 / total_type * 100
    top3_accuracy_type = correct_type_top3 / total_type * 100

    print(f"Top-1 Accuracy for Spacing Type: {top1_accuracy_type:.2f}%")
    print(f"Top-3 Accuracy for Spacing Type: {top3_accuracy_type:.2f}%")


if __name__ == "__main__":
    # create_datasets()
    data = preprocessing.load_data("data.jsonl")

    input, output, vocab_stoi, vocab_itos = get_input_output_training(data)
    input = [torch.tensor(inp) for inp in input]
    output = [torch.tensor(out) for out in output]
    input = pad_sequence(input, batch_first=True, padding_value=0)
    output = pad_sequence(output, batch_first=True, padding_value=-1)
    training_inputs, validation_inputs, training_outputs, validation_outputs = train_test_split(input, output, test_size=0.2, random_state=27)

    vocab_size = len(vocab_stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_model = TokenSpacingModel(vocab_size).to(device)
    loss_fn_type = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(code_model.parameters(), lr=0.001)

    batch_size = 2
    epochs = 10

    print("--- TRAINING STARTED ---")
    train(code_model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, optimizer, device)
    print("--- EVALUATION STARTED ---")
    evaluate(code_model, validation_inputs, validation_outputs, batch_size, device)
