"""Utility functions for training and building models for text and audio
confidence classification.
"""
import os
import pandas as pd
import itertools
from prettytable import PrettyTable
import json
import ast
import torch
import numpy as np
import random
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

num_gpus = torch.cuda.device_count()


class TextDataset(torch.utils.data.Dataset):
    """
    Prepare the dataset according to its attributes.
    """

    def __init__(self, df, tokenizer):
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


def train_text(model, tokenizer, train_data, val_data, learning_rate, epochs):
    """Train the model based on extracted text."""
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = TextDataset(train, tokenizer), TextDataset(val, tokenizer)

    #
    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=10, shuffle=True, drop_last=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=10, num_workers=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("Using cuda!")
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
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
    test = test_data.reset_index(drop=True)
    test = TextDataset(test)

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


def load_audio_and_score_from_folder(folder_path_dir):
    """
    Load the confidence score and audio array from the csv files.
    :param home_dir: Primary directory.
    :param folder_path_list: Path of the folder of csvs.
    :return: result_df: Pandas dataframe wit columns audio_array and score.
    """
    audio_list = []
    score_list = []
    max_length = 0
    for filename in tqdm(os.listdir(folder_path_dir)):
        total_df = pd.read_csv(
            os.path.join(folder_path_dir, filename), encoding="utf-8", low_memory=False
        )
        print(filename)
        try:
            # Convert to numpy array
            curr_audio_data = total_df["audio_array"].to_list()
            # If list contains element of type string
            if not all(isinstance(i, float) for i in curr_audio_data):
                print("Found wrong data type!")
                # Decode to float using jason
                curr_audio_data = json.loads(curr_audio_data[0])
                curr_audio_data = [float(elem) for elem in curr_audio_data]
                print(type(curr_audio_data[0]))
            audio_list.append(curr_audio_data)
            score_list.append(random.choice(range(1, 10, 1)))
            # Update max length if a longer audio occurs
            if len(total_df["audio_array"]) > max_length:
                max_length = len(total_df["audio_array"])
        except:
            print("Error in parsing! File name = " + filename)
            continue

    print(len(audio_list))
    print(len(score_list))
    result_df = pd.DataFrame(
        np.column_stack([audio_list, score_list]), columns=["audio_array", "score"]
    )
    # result = feature_extractor(audio_list, sampling_rate=16000, padding=True)
    # result["labels"] = score_list
    save_path = os.path.join(folder_path_dir, "test_model.json")
    result_df.to_csv(save_path, index=False)
    # json_file = result_df.to_json()
    # with open(save_path, 'w') as outfile:
    #     json.dump(json_file, outfile, indent=4)
    return result_df


class AudioDataset(torch.utils.data.Dataset):
    """
    Prepare the audio dataset according to its attributes.
    """

    def __init__(self, df, feature_extractor):
        self.labels = df["score"]
        self.df = df
        self.feature_extractor = feature_extractor
        self.audios = []
        self.extract_audio_features()

    def extract_audio_features(self):
        audios = []
        for audio in self.df["audio_array"]:
            # audio_lst = json.loads(audio[0])
            # audio = [float(ele) for ele in audio[0]]
            extracted_tensor = self.feature_extractor(
                audio,
                sampling_rate=16000,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            audios.append(extracted_tensor)
        self.audios = audios

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_audios(self, idx):
        # Fetch a batch of inputs
        return self.audios[idx]

    def __getitem__(self, idx):
        batch_audios = self.get_batch_audios(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_audios, batch_y


class AudioDatasetNew(torch.utils.data.Dataset):
    """
    Prepare the audio dataset according to its attributes.
    """

    def __init__(self, df):
        self.labels = df["score"]
        self.audios = df["audio_array"]
        self.audios_list = self.audios.tolist()
        print(type(self.audios_list[0]))
        self.max_length = 0
        self.padded_audio = None

    def find_max_array_length(self):
        list_len = [len(i) for i in self.audios_list]
        max_length = max(list_len)
        self.max_length = max_length

    def pad_audio(self):
        pad_token = float(0)
        padded = zip(*itertools.zip_longest(*self.audios_list, fillvalue=pad_token))
        self.padded_audio = padded

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_audios(self, idx):
        # Fetch a batch of inputs
        return self.audios[idx]

    def __getitem__(self, idx):
        batch_audios = self.get_batch_audios(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_audios, batch_y


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_audio(model, feature_extractor, train_data, val_data, learning_rate, epochs):
    """Train the model based on extracted audio features."""
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = (
        AudioDataset(train, feature_extractor),
        AudioDataset(val, feature_extractor),
    )
    # train, val = (
    #     AudioDatasetNew(train),
    #     AudioDataset(val),
    # )

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1, num_workers=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        # criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            print(type(train_input))
            mask = train_input["attention_mask"].to(device)
            input_values = train_input["input_values"].squeeze(1).to(device)

            output = model(input_values, mask)

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
                input_id = val_input["input_values"].squeeze(1).to(device)

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

def train_audio_raw(model, train_data, val_data, learning_rate, epochs):
    """Train the model based on extracted audio only."""
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = (
        AudioDatasetNew(train),
        AudioDatasetNew(val),
    )


    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1, num_workers=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        # criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            print(type(train_input))
            mask = train_input["attention_mask"].to(device)
            input_values = train_input["input_values"].squeeze(1).to(device)

            output = model(input_values, mask)

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
                input_id = val_input["input_values"].squeeze(1).to(device)

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

def evaluate_audio(model, test_data):
    """Evaluate accuracy for audio data."""
    test = test_data.reset_index(drop=True)
    test = AudioDataset(test)

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
            input_id = test_input["input_values"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
