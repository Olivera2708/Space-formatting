import preprocessing
from sklearn.model_selection import train_test_split
from format_java import format_java_code, tokenize_and_create_input_output, tokenize_and_classify, create_vocab
from model import TokenSpacingModel
import torch
import random
from torch.utils.data import DataLoader, TensorDataset

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

import torch

def train(model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, loss_fn_length, optimizer, device):
    model.to(device)
    
    dataset = TensorDataset(training_inputs, training_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_batch, output_batch in dataloader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            optimizer.zero_grad()
            spacing_type_pred, length_of_spacing_pred = model(input_batch)

            spacing_output = output_batch[:, 0][:-1]
            length_output = output_batch[:, 1][:-1].view(-1, 1).float()

            loss_type = loss_fn_type(spacing_type_pred, spacing_output)
            loss_length = loss_fn_length(length_of_spacing_pred, length_output)
            total_loss_batch = loss_type + loss_length
            total_loss += total_loss_batch.item()

            total_loss_batch.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate(model, eval_inputs, eval_outputs, batch_size, device, length_threshold=0.1):
    model.eval()
    dataset = TensorDataset(eval_inputs, eval_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    correct_combined = 0  # Correct predictions for both type and length
    total_samples = 0  # Total number of samples

    correct_type = 0  # Correct predictions for spacing type
    total_type = 0  # Total number of samples for spacing type
    
    correct_length = 0  # Correct predictions for spacing length (within threshold)
    total_length = 0  # Total number of samples for spacing length
    
    with torch.no_grad():
        for input_batch, output_batch in dataloader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            # Get model predictions
            spacing_type_pred, length_of_spacing_pred = model(input_batch)

            # Calculate Top-1 accuracy for spacing type (classification task)
            _, top1_pred_type = torch.max(spacing_type_pred, dim=1)
            correct_type += torch.sum(top1_pred_type == output_batch[:, 0][:-1]).item()
            total_type += len(input_batch)

            # Calculate accuracy for spacing length (regression task)
            correct_length += torch.sum(torch.abs(length_of_spacing_pred.squeeze() - output_batch[:, 1][:-1]) < length_threshold).item()
            total_length += len(input_batch)

            # Combined accuracy for both spacing type and spacing length
            combined_correct = torch.sum(
                (top1_pred_type == output_batch[:, 0][:-1]) & 
                (torch.abs(length_of_spacing_pred.squeeze() - output_batch[:, 1][:-1]) < length_threshold)
            ).item()

            correct_combined += combined_correct
            total_samples += len(input_batch)

    # Calculate and print individual accuracies
    top1_accuracy_type = correct_type / total_type * 100
    print(f"Top-1 Accuracy for Spacing Type: {top1_accuracy_type:.2f}%")

    top1_accuracy_length = correct_length / total_length * 100
    print(f"Top-1 Accuracy for Spacing Length: {top1_accuracy_length:.2f}%")

    # Calculate and print combined accuracy for both type and length
    combined_accuracy = correct_combined / total_samples * 100
    print(f"Combined Accuracy (Spacing Type + Spacing Length): {combined_accuracy:.2f}%")


if __name__ == "__main__":
    # create_datasets()
    data = preprocessing.load_data("data.jsonl")

    input, output, vocab_stoi, vocab_itos = get_input_output_training(data)
    input = torch.tensor(input, dtype=torch.long)
    output = torch.tensor(output, dtype=torch.long)
    training_inputs, validation_inputs, training_outputs, validation_outputs = train_test_split(input, output, test_size=0.2, random_state=27)

    vocab_size = len(vocab_stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    code_model = TokenSpacingModel(vocab_size).to(device)
    loss_fn_type = torch.nn.CrossEntropyLoss()
    loss_fn_length = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(code_model.parameters(), lr=0.001)

    batch_size = 32
    epochs = 10

    # Train and evaluate the model
    train(code_model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, loss_fn_length, optimizer, device)
    evaluate(code_model, validation_inputs, validation_outputs, batch_size, device)
