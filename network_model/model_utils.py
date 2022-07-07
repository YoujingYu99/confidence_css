"""Utility functions for training and building models for text and audio
confidence classification.
"""
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


class TextDataset(torch.utils.data.Dataset):
    """
    Prepare the dataset according to its attributes.
    """

    def __init__(self, df):
        self.labels = df["score"]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["sentence"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def train_text(model, train_data, val_data, learning_rate, epochs):
    """Train the model based on extracted text."""
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = TextDataset(train), TextDataset(val)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1000)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print('Using cuda!')
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        print(epoch_num)
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_text(model, test_data):
    """Evaluate accuracy for text data."""
    test = TextDataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
