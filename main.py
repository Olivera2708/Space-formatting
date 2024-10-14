import preprocessing
from sklearn.model_selection import train_test_split
from format_java import format_java_code, tokenize_and_create_input_output, tokenize_and_classify, create_vocab
from model import CodeFormatterModel
import torch
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
from torch.utils.data import DataLoader, TensorDataset

def train(model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, loss_fn_length, optimizer, device):
    train_data = TensorDataset(training_inputs, training_outputs)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss_type = 0
        epoch_loss_length = 0

        for input_batch, output_batch in train_loader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            optimizer.zero_grad()
            
            spacing_type, length_of_spacing = model(input_batch)

            print(spacing_type)
            print(length_of_spacing)

            print(output_batch[:, 0])
            print(output_batch[:, 1])

            loss_type = loss_fn_type(spacing_type, output_batch[:, 0])  # Classification loss
            loss_length = loss_fn_length(length_of_spacing, output_batch[:, 1])  # Regression loss

            # Total loss
            loss = loss_type + loss_length
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Accumulate the loss for this epoch
            epoch_loss_type += loss_type.item()
            epoch_loss_length += loss_length.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss Type: {epoch_loss_type / len(train_loader):.4f}, Loss Length: {epoch_loss_length / len(train_loader):.4f}")



def eval(model, validation_inputs, validation_outputs, batch_size, loss_fn_type, loss_fn_length, device):
    model.eval()
    total_loss_type = 0
    total_loss_length = 0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    val_data = TensorDataset(validation_inputs, validation_outputs)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            # Forward pass
            spacing_type, length_of_spacing = model(*input_batch)

            # Calculate the individual losses
            loss_type = loss_fn_type(spacing_type, output_batch[:, 0])
            loss_length = loss_fn_length(length_of_spacing, output_batch[:, 1])

            total_loss_type += loss_type.item()
            total_loss_length += loss_length.item()

            # Top-1 and Top-3 accuracy
            _, top1_pred = torch.max(spacing_type, dim=-1)
            top3_pred = torch.topk(spacing_type, 3, dim=-1).indices

            correct_top1 += (top1_pred == output_batch[:, 0]).sum().item()
            correct_top3 += (output_batch[:, 0].unsqueeze(-1) == top3_pred).any(dim=-1).sum().item()

            total += output_batch.numel()

    avg_loss_type = total_loss_type / len(val_loader)
    avg_loss_length = total_loss_length / len(val_loader)
    top1_accuracy = correct_top1 / total * 100
    top3_accuracy = correct_top3 / total * 100

    print(f"Validation Loss Type: {avg_loss_type:.4f}, Loss Length: {avg_loss_length:.4f}, Top-1 Accuracy: {top1_accuracy:.2f}%, Top-3 Accuracy: {top3_accuracy:.2f}%")

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
    code_model = CodeFormatterModel(vocab_size).to(device)
    loss_fn_type = torch.nn.CrossEntropyLoss()
    loss_fn_length = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(code_model.parameters(), lr=0.001)

    batch_size = 32
    epochs = 10

    # Train and evaluate the model
    train(code_model, epochs, training_inputs, training_outputs, batch_size, loss_fn_type, loss_fn_length, optimizer, device)
    # eval(code_model, validation_inputs, validation_outputs, batch_size, vocab_size, vocab_itos, loss_fn, device)
