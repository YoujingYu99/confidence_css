"""Utility functions for training and building models for text and audio
confidence classification.

----------------------------------
Class TextDataset: Class that handles the preparation of text for training.
Class AudioDataset: Class that handles the preparation of audio for training.
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import pandas as pd
import math
from prettytable import PrettyTable
import json
import numpy as np
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import *

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
        """Get labels of each audio."""
        return self.labels

    def __len__(self):
        """Get the total number of audios in the dataset."""
        return len(self.labels)

    def get_batch_labels(self, idx):
        """Fetch a batch of labels."""
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """Fetch a batch of inputs."""
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def categorise_score(score):
    """
    Categorise the confidnece scores into 5 categories.
    :param score: Raw score input by user.
    :return: Categorised score.
    """
    if score == 5:
        score_cat = 4
    else:
        score_cat = math.floor(score)

    return score_cat


def test_accuracy(output, actual):
    """
    Testfy whether the output is accurate.
    :param output: Score tensor output by model.
    :param actual: Actual score tensor.
    :return: Number of accurate predicitons
    """
    output_list = output.tolist()
    actual_list = actual.tolist()
    count = 0
    for i in range(len(output_list)):
        if actual_list[i] * 0.8 <= output_list[i] <= actual_list[i] * 1.2:
            count += 1
    return count


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        # tolerance is the number of epochs to continue after deemed saturated
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        print("val loss - train loss", validation_loss - train_loss)
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def train_text(
    model,
    tokenizer,
    train_data,
    val_data,
    learning_rate,
    weight_decay,
    epochs,
    batch_size,
    num_workers,
):
    """
    Train the model based on extracted text.
    :param model: Deep learning model for the text.
    :param tokenizer: Pre-trained transformer to tokenize the text.
    :param train_data: Dataframe to be trained.
    :param val_data: Dataframe to be evaluated.
    :param learning_rate: Parameter; rate of learning.
    :param weight_decay: Rate of decay (l2).
    :param epochs: Number of epochs to be trained.
    :param batch_size: Number of batches.
    :return: Training and evaluation accuracies.
    """
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = TextDataset(train, tokenizer), TextDataset(val, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(tolerance=5, min_delta=1)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            output = model(input_id, mask)
            output = output.flatten()
            batch_loss = criterion(output.float(), train_label.float())
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, train_label)
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)
                output = model(input_id, mask)
                output = output.flatten()
                batch_loss = criterion(output.float(), val_label.float())
                total_loss_val += batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # Define accuracy as within 10% of the true label
                acc = test_accuracy(output, val_label)
                total_acc_val += acc
            # early stopping
            early_stopping(
                total_loss_train / len(train_data), total_loss_val / len(val_data)
            )

            # Append to list
            train_loss_list.append(total_loss_train / len(train_data))
            train_acc_list.append(total_acc_train / len(train_data))
            val_loss_list.append(total_loss_val / len(val_data))
            val_acc_list.append(total_acc_val / len(val_data))

            # Generate plots
            plot_name = "text_lstm_"
            gen_acc_plots(train_acc_list, val_acc_list, plot_name)
            gen_loss_plots(train_loss_list, val_loss_list, plot_name)
            save_training_results(
                train_loss_list, train_acc_list, val_loss_list, val_acc_list, plot_name
            )

            if early_stopping.early_stop:
                print("We are at epoch:", epoch_num)
                break

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_text(model, test_data, tokenizer, batch_size):
    """
    Evaluate accuracy for the model on text data.
    :param model: Model to be used for deep learning.
    :param test_data: Dataframe to be tested.
    :param tokenizer: Pre-trained transformer to tokenize the text.
    :param batch_size: Number of batches.
    :return: Test Accuracies.
    """
    test = test_data.reset_index(drop=True)
    test = TextDataset(test, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, drop_last=True, pin_memory=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    total_acc_test = 0
    with torch.no_grad():
        model.eval()
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)
            output = output.flatten()

            # acc = (output.argmax(dim=1) == test_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, test_label)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


def load_audio_and_score_from_folder(folder_path_dir, file_type, save_to_single_csv):
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
        if (
            filename != "audio_only_all_model.csv"
            and filename != "select_features_all_model.csv"
        ):
            try:
                total_df = pd.read_csv(
                    os.path.join(folder_path_dir, filename),
                    encoding="utf-8",
                    low_memory=False,
                    delimiter=",",
                )
            except Exception as e:
                print("Error in parsing! File name = " + filename)
                print(e)
                continue

            try:
                # Convert to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = json.loads(curr_audio_data[0])
                    curr_audio_data = [float(elem) for elem in curr_audio_data]
                    print(type(curr_audio_data[0]))
                audio_list.append(curr_audio_data)
                score_list.append(random.choice(range(1, 5, 1)))
                # Update max length if a longer audio occurs
                if len(total_df["audio_array"]) > max_length:
                    max_length = len(total_df["audio_array"])
            except Exception as e:
                print("Error in parsing! File name = " + filename)
                print(e)
                continue

    print(len(audio_list))
    print(len(score_list))
    result_df = pd.DataFrame(
        np.column_stack([audio_list, score_list]), columns=["audio_array", "score"]
    )
    if save_to_single_csv:
        ## Save all data into a single csv file.
        if file_type == "audio_only":
            save_path = os.path.join(folder_path_dir, "audio_only_all_model.csv")
        elif file_type == "select_features":
            save_path = os.path.join(folder_path_dir, "select_features_all_model.csv")
        result_df.to_csv(save_path, index=False)
    return result_df


def load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
):
    """
    Load the audio arrays and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :return: Dataframe of audio arrays and average score.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    # Initialise empty lists
    audio_list = []
    score_list = []
    # Find the feature csv locally
    for index, row in results_df.iterrows():
        audio_url = row["audio_url"]
        # https://extractedaudio.s3.eu-west-2.amazonaws.com/5/C_show_5CnDmMUG0S5bSSw612fs8C_3fxFPVGSzFLKf5iyg5rWCa_1917.0.mp3
        folder_number = audio_url.split("/")[-2]
        segment_name = audio_url.split("/")[-1][:-4]
        total_df_path = os.path.join(
            home_dir,
            "data_sheets",
            "features",
            str(folder_number),
            segment_name + ".csv",
        )
        # Only proceed if file exists
        if os.path.isfile(total_df_path):
            total_df = pd.read_csv(total_df_path, encoding="utf-8", dtype="unicode")
            score_list.append(row["average"] - 2.5)
            try:
                # Convert audio to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    # print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = [json.loads(i) for i in curr_audio_data]
                audio_list.append(curr_audio_data)

                # # Convert to list
                # curr_audio_data = total_df["audio_array"].to_list()
                # # If list contains element of type string
                # if not all(isinstance(i, float) for i in curr_audio_data):
                #     print("Found wrong data type!")
                #     # Decode to float using json
                #     curr_audio_data = json.loads(curr_audio_data[0])
                #     curr_audio_data = [float(elem) for elem in curr_audio_data]
                #     print(type(curr_audio_data[0]))
                # audio_list.append(curr_audio_data)
            except Exception as e:
                print("Error in parsing! File name = " + total_df_path)
                print(e)
                continue

    print(len(audio_list))
    print(len(score_list))
    result_df = pd.DataFrame(
        np.column_stack([audio_list, score_list]), columns=["audio_array", "score"]
    )
    if save_to_single_csv:
        ## Save all data into a single csv file.
        save_path = os.path.join(
            home_dir, "data_sheets", "audio_only_crowd_all_model.csv"
        )
        result_df.to_csv(save_path, index=False)
    return result_df


def load_text_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
):
    """
    Load the text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :return: Dataframe of text and average score.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    # Initialise empty lists
    text_list = []
    score_list = []
    # Find the feature csv locally
    for index, row in results_df.iterrows():
        audio_url = row["audio_url"]
        # https://extractedaudio.s3.eu-west-2.amazonaws.com/5/C_show_5CnDmMUG0S5bSSw612fs8C_3fxFPVGSzFLKf5iyg5rWCa_1917.0.mp3
        folder_number = audio_url.split("/")[-2]
        segment_name = audio_url.split("/")[-1][:-4]
        all_features_csv_path = os.path.join(
            home_dir,
            "data_sheets",
            "features",
            str(folder_number),
            segment_name + ".csv",
        )
        # Only proceed if file exists
        if os.path.isfile(all_features_csv_path):
            try:
                select_features_df = pd.read_csv(
                    all_features_csv_path, encoding="utf-8", dtype="unicode"
                )
                # Conver to numpy integer type
                score_list.append(row["average"] - 2.5)
                # Convert to list
                curr_text_data = select_features_df["text"].to_list()[0]
                # print("curr text data", curr_text_data)
                # print(type(curr_text_data))
                text_list.append([curr_text_data])
            except Exception as e:
                print("Error in parsing! File name = " + all_features_csv_path)
                print(e)
                continue

    print(len(text_list))
    print(len(score_list))
    result_df = pd.DataFrame(
        np.column_stack([text_list, score_list]), columns=["sentence", "score"]
    )
    result_df["score"] = result_df["score"].astype(float)
    if save_to_single_csv:
        ## Save all data into a single csv file.
        save_path = os.path.join(
            home_dir, "data_sheets", "text_only_crowd_all_model.csv"
        )
        result_df.to_csv(save_path, index=False)
    return result_df


def load_audio_text_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
):
    """
    Load the audio arrays, text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :return: Dataframe of audio arrays, text and average score.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    # Initialise empty lists
    audio_list = []
    text_list = []
    score_list = []
    # Find the feature csv locally
    for index, row in results_df.iterrows():
        audio_url = row["audio_url"]
        # https://extractedaudio.s3.eu-west-2.amazonaws.com/5/C_show_5CnDmMUG0S5bSSw612fs8C_3fxFPVGSzFLKf5iyg5rWCa_1917.0.mp3
        folder_number = audio_url.split("/")[-2]
        segment_name = audio_url.split("/")[-1][:-4]
        total_df_path = os.path.join(
            home_dir,
            "data_sheets",
            "features",
            str(folder_number),
            segment_name + ".csv",
        )
        # Only proceed if file exists
        if os.path.isfile(total_df_path):
            total_df = pd.read_csv(total_df_path, encoding="utf-8", dtype="unicode")
            score_list.append(row["average"] - 2.5)
            try:
                # Convert audio to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    # print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = [json.loads(i) for i in curr_audio_data]
                audio_list.append(curr_audio_data)

                # Convert text to list
                curr_text_data = total_df["text"].to_list()[0]
                text_list.append([curr_text_data])
            except Exception as e:
                print("Error in parsing! File name = " + total_df_path)
                print(e)
                continue

    print(len(audio_list))
    print(len(text_list))
    print(len(score_list))
    result_df = pd.DataFrame(
        np.column_stack([audio_list, text_list, score_list]),
        columns=["audio_array", "sentence", "score"],
    )
    if save_to_single_csv:
        ## Save all data into a single csv file.
        save_path = os.path.join(
            home_dir, "data_sheets", "audio_text_crowd_all_model.csv"
        )
        result_df.to_csv(save_path, index=False)
    return result_df


def load_select_features_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path
):
    """
    Load the text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :return: A dictoinary of dataframes.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    all_dict = {}
    # Initialise empty lists
    # Find the feature csv locally
    for index, row in results_df.iterrows():
        audio_url = row["audio_url"]
        # https://extractedaudio.s3.eu-west-2.amazonaws.com/5/C_show_5CnDmMUG0S5bSSw612fs8C_3fxFPVGSzFLKf5iyg5rWCa_1917.0.mp3
        folder_number = audio_url.split("/")[-2]
        segment_name = audio_url.split("/")[-1][:-4]
        select_features_csv_path = os.path.join(
            home_dir,
            "data_sheets",
            "features",
            str(folder_number),
            segment_name + ".csv",
        )
        # Only proceed if file exists
        if os.path.isfile(select_features_csv_path):
            # print(select_features_csv_path)
            # select_features_df = pd.read_csv(select_features_csv_path, encoding="utf-8", dtype="unicode")
            score = [row["average"] - 2.5]
            # select_features_csv_path = os.path.join(home_dir, "data_sheets",
            #                                    "features",
            #                                    str(folder_number),
            #                                    segment_name + ".csv")
            # select_features_df = pd.read_csv(select_features_csv_path)
            try:
                select_features_df = pd.read_csv(
                    select_features_csv_path, encoding="utf-8", dtype="unicode"
                )
                # Remove the audio array
                select_features_df.drop("audio_array", axis=1, inplace=True)
                new_length = select_features_df["energy"].count()
                select_features_df_new = select_features_df.iloc[:new_length].copy()
                # Extend the list with 0 to match size of other columns
                score.extend([0] * (select_features_df_new.shape[0] - 1))
                # Add the audio score to the dataframe
                select_features_df_new["score"] = score
                # Zero pad the dataframe
                select_features_df_new.fillna(0)
                all_dict[audio_url] = select_features_df_new
            except Exception as e:
                print("Error in parsing! File name = " + select_features_csv_path)
                print(e)
                continue
    return all_dict


def load_select_features_and_score_from_crowdsourcing_results_selective(
    home_dir, crowdsourcing_results_df_path, features_to_use
):
    """
    Load the text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param features_to_use: Features to be passed into the model.
    :return: A dictoinary of dataframes.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    all_dict = {}
    # Initialise empty lists
    # Find the feature csv locally
    for index, row in results_df.iterrows():
        audio_url = row["audio_url"]
        # https://extractedaudio.s3.eu-west-2.amazonaws.com/5/C_show_5CnDmMUG0S5bSSw612fs8C_3fxFPVGSzFLKf5iyg5rWCa_1917.0.mp3
        folder_number = audio_url.split("/")[-2]
        segment_name = audio_url.split("/")[-1][:-4]
        all_features_csv_path = os.path.join(
            home_dir,
            "data_sheets",
            "features",
            str(folder_number),
            segment_name + ".csv",
        )
        # Only proceed if file exists
        if os.path.isfile(all_features_csv_path):
            score = [row["average"] - 2.5]
            try:
                all_features_df = pd.read_csv(
                    all_features_csv_path, encoding="utf-8", dtype="unicode"
                )
                # select the features to be copied
                select_features_df = all_features_df[features_to_use].copy()
                new_length = select_features_df["energy"].count()
                select_features_df_new = select_features_df.iloc[:new_length].copy()
                # Extend the list with 0 to match size of other columns
                score.extend([0] * (select_features_df_new.shape[0] - 1))
                # Add the audio score to the dataframe
                select_features_df_new["score"] = score
                # Zero pad the dataframe
                select_features_df_new.fillna(0)
                all_dict[audio_url] = select_features_df_new
            except Exception as e:
                print("Error in parsing! File name = " + all_features_csv_path)
                print(e)
                continue
    return all_dict


def get_num_rows(dictionary):
    """
    Get the maximum number of rows in the df in a dictionary.
    :param dictionary: A dictionary of dataframes.
    :return:
    """
    max_row_length = 0
    for audio_name, select_features_df in dictionary.items():
        # Update max row length
        if select_features_df.shape[0] > max_row_length:
            max_row_length = select_features_df.shape[0]
    return max_row_length


class SelectFeaturesDataset(torch.utils.data.Dataset):
    """
    Prepare features and lables separately as dataset.
    """

    def __init__(self, dict, num_rows, num_columns):
        self.dict = dict
        self.labels = self.get_labels()
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.features_dict = self.get_features_dict()
        self.features_list = list(self.features_dict.items())

    def get_labels(self):
        """
        Get the labels of the audios from the features df.
        :return: A list of scores of the audios
        """
        score_list = []
        for audio_name, select_features_df in self.dict.items():
            score = select_features_df["score"][0]
            score_list.append(score)
        return score_list

    def get_features_dict(self):
        """
        Get the features as training input and pad to same length.
        :return: A dictionary of tensors with only features.
        """
        features_only_dict = {}
        features_only_dict_new = {}

        for audio_name, select_features_df in self.dict.items():
            # Remove the scores
            select_features_df.drop("score", axis=1, inplace=True)
            features_only_dict[audio_name] = select_features_df

        # Pad to zero so same length
        for audio_name, select_features_df in features_only_dict.items():
            # Zero pad dataframe to same num of rows defined
            num_rows_to_append = self.num_rows - select_features_df.shape[0]
            select_features_df = select_features_df.append(
                [[0] for _ in range(num_rows_to_append)], ignore_index=True
            )
            # Replace nan with 0
            select_features_df = select_features_df.fillna(0)
            # Deal with the situation where an additional column is added at the end
            if select_features_df.shape[1] != self.num_columns:
                print("this wrong df no. columns", select_features_df.shape[1])
                select_features_df.drop(
                    columns=select_features_df.columns[-1], axis=1, inplace=True
                )
            # Append to the new dictionary
            features_only_dict_new[audio_name] = torch.tensor(
                select_features_df.values.astype(np.float32)
            )

        return features_only_dict_new

    def classes(self):
        """Get labels of each audio."""
        return self.labels

    def __len__(self):
        """Get the total number of audios in the dataset."""
        return len(self.labels)

    def get_batch_labels(self, idx):
        """Fetch a batch of labels."""
        return np.array(self.labels[idx])

    def get_batch_features(self, idx):
        """Fetch a batch of inputs."""
        return self.features_list[idx]

    def __getitem__(self, idx):
        batch_features = self.get_batch_features(idx)
        # batch_features is a tuple of (audio_url, tensor)
        batch_y = self.get_batch_labels(idx)

        return batch_features, batch_y


def train_select_features(
    train_data,
    val_data,
    learning_rate,
    epochs,
    batch_size,
    num_workers,
    num_of_rows,
    num_of_columns,
):
    """
    Train the model based on extracted text.
    :param train_data: Dict of Dataframe to be trained.
    :param val_data: Dict of Dataframe to be evaluated.
    :param learning_rate: Parameter; rate of learning.
    :param epochs: Number of epochs to be trained.
    :param batch_size: Number of batches.
    :param num_of_rows: Maximum row number in the whole dataset.
    :param num_of_columns: Number of columns in final df.
    :return: Training and evaluation accuracies.
    """
    train, val = train_data, val_data
    train, val = (
        SelectFeaturesDataset(train, num_of_rows, num_of_columns),
        SelectFeaturesDataset(val, num_of_rows, num_of_columns),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()
    model = SelectFeaturesClassifier(
        num_rows=train.num_rows, num_columns=train.num_columns
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            input_features = train_input[1]
            output = model(input_features)
            output = output.flatten()
            batch_loss = criterion(output.float(), train_label.float())
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, train_label)
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                input_features = val_input[1]
                output = model(input_features)
                output = output.flatten()
                batch_loss = criterion(output.float(), val_label.float())
                total_loss_val += batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # Define accuracy as within 10% of the true label
                acc = test_accuracy(output, val_label)
                total_acc_val += acc

            # early stopping
            early_stopping(total_loss_train, total_loss_val)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch_num)
                break

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_select_features(test_data, batch_size, num_of_rows, num_of_columns):
    """
    Evaluate accuracy for the model on text data.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param num_of_rows: Maximum row number in the whole dataset.
    :param num_of_columns: Number of columns in final df.
    :return: Test Accuracies.
    """
    test = test_data
    test = SelectFeaturesDataset(test, num_of_rows, num_of_columns)

    test_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, drop_last=True, pin_memory=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = SelectFeaturesClassifier(
        num_rows=test.num_rows, num_columns=test.num_columns
    )

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    total_acc_test = 0
    with torch.no_grad():
        model.eval()
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            input_features = test_input[1]
            output = model(input_features)
            output = output.flatten()

            # acc = (output.argmax(dim=1) == test_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, test_label)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


class AudioDataset(torch.utils.data.Dataset):
    """
    Vectorise the audio arrays using the Wav2Vec transformer and prepare
    as dataset.
    """

    def __init__(self, df, feature_extractor, vectorise):
        self.df = df
        self.labels = df["score"]
        self.audio_series = df["audio_array"]
        self.audios_list = self.audio_series.tolist()

        self.feature_extractor = feature_extractor
        self.max_length = 0
        self.audios = None

        if vectorise:
            self.extract_audio_features()
        else:
            # Get padded audio
            self.find_max_array_length()
            self.pad_audio()

    def extract_audio_features(self):
        """
        Extract audio features using preloaded feature extractor.
        :return: Assign a list of tensors to self.audios.
        """
        audios = []
        for audio in self.audio_series:
            # Extract the features
            extracted_tensor = self.feature_extractor(
                audio,
                sampling_rate=16000,
                padding="max_length",
                truncation=True,
                max_length=200000,
                return_tensors="pt",
            )
            audios.append(extracted_tensor)
        # Reassign vectors as audios
        self.audios = audios

    def find_max_array_length(self):
        """
        Find maximum length of the inidividual audio arrays.
        :return: Assign an integer of maximum length to self.max_length.
        """
        list_len = [len(i) for i in self.audios_list]
        max_length = max(list_len)
        self.max_length = max_length

    def pad_audio(self):
        """
        Pad shorter audios with 0 to make all audio arrays equal lengths.
        :return: Assign a list of assrays to self.audios.
        """
        new_list = []
        for audio_array in self.audios_list:
            padded_audio_array = np.pad(
                audio_array, (0, self.max_length - len(audio_array)), "constant"
            )
            new_list.append(padded_audio_array)
        # audios is a list of arrays.
        self.audios = new_list

    def classes(self):
        """Get labels of each audio."""
        return self.labels

    def __len__(self):
        """Get the total number of audios in the dataset."""
        return len(self.labels)

    def get_batch_labels(self, idx):
        """Fetch a batch of labels."""
        return np.array(self.labels[idx])

    def get_batch_audios(self, idx):
        """Fetch a batch of inputs."""
        return self.audios[idx]

    def __getitem__(self, idx):
        batch_audios = self.get_batch_audios(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_audios, batch_y


def count_parameters(model):
    """
    Print parameter names and layers and count the total number of
    parameters.
    :param model: The name of the deep learning model.
    :return: The total number of parameters as integer.
    """
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


def train_audio(
    model,
    feature_extractor,
    train_data,
    val_data,
    learning_rate,
    weight_decay,
    epochs,
    batch_size,
    vectorise,
    num_workers,
):
    """
    Train the model based on extracted audio vectors.
    :param model: Deep learning model for the audio training.
    :param feature_extractor: Pre-trained transformer to extract audio features.
    :param train_data: Dataframe to be trained.
    :param val_data: Dataframe to be evaluated.
    :param learning_rate: Parameter; rate of learning.
    :param weight_decay: Weight of decay; l2 regularisation.
    :param epochs: Number of epochs to be trained.
    :param batch_size: Number of batches.
    :param vectorise: If vectorised, use transformers to tokenize audios.
    :return: Training and evaluation accuracies.
    """
    # Prepare data into dataloader
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = (
        AudioDataset(train, feature_extractor, vectorise),
        AudioDataset(val, feature_extractor, vectorise),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(tolerance=5, min_delta=1)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            # mask = train_input["attention_mask"].to(device)
            if vectorise:
                input_values = train_input["input_values"].squeeze(1).to(device)
            else:
                input_values = train_input.squeeze(1).to(device, dtype=torch.float)
                print("input size", input_values.size())

            optimizer.zero_grad()
            output = model(input_values)
            output = output.flatten()
            batch_loss = criterion(output.float(), train_label.float())
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, train_label)
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                # mask = val_input["attention_mask"].to(device)
                if vectorise:
                    input_values = train_input["input_values"].squeeze(1).to(device)
                else:
                    # print("train input type", type(train_input))
                    # print(train_input)
                    # print(type(train_input["input_values"]))
                    # input_values = train_input["input_values"].to(device, dtype=torch.float)
                    # print(input_values.size())
                    input_values = train_input.to(device, dtype=torch.float)

                output = model(input_values)
                output = output.flatten()
                batch_loss = criterion(output.float(), train_label.float())
                total_loss_val += batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # Define accuracy as within 10% of the true label
                acc = test_accuracy(output, val_label)
                total_acc_val += acc

        # early stopping
        early_stopping(
            total_loss_train / len(train_data), total_loss_val / len(val_data)
        )

        # Append to list
        train_loss_list.append(total_loss_train / len(train_data))
        train_acc_list.append(total_acc_train / len(train_data))
        val_loss_list.append(total_loss_val / len(val_data))
        val_acc_list.append(total_acc_val / len(val_data))
        # Generate plots
        plot_name = "audio_lstm_"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        save_training_results(
            train_loss_list, train_acc_list, val_loss_list, val_acc_list, plot_name
        )

        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num)
            break

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_audio(model, test_data, batch_size, feature_extractor, vectorise):
    """
    Evaluate accuracy for the model on vectorised audio data.
    :param model: Model to be used for deep learning.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param feature_extractor: Pre-trained transformer to extract audio features.
    :param vectorise: If vectorised, use transformers to tokenize audios.
    :return: Test Accuracies.
    """
    test = test_data.reset_index(drop=True)
    test = AudioDataset(test, feature_extractor, vectorise)
    test_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, drop_last=True, pin_memory=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    total_acc_test = 0
    with torch.no_grad():
        model.eval()
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            # mask = test_input["attention_mask"].to(device)
            if vectorise:
                input_values = test_input["input_values"].squeeze(1).to(device)
            else:
                input_values = torch.cat(test_input).to(device, dtype=torch.float)

            output = model(input_values)
            output = output.flatten()

            # acc = (output.argmax(dim=1) == test_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, test_label)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


class AudioTextDataset(torch.utils.data.Dataset):
    """
    Vectorise the audio arrays and text using the transformers and prepare
    as dataset.
    """

    def __init__(self, df, audio_feature_extractor, text_tokenizer, vectorise):
        self.df = df
        self.labels = df["score"]

        # Get audio
        self.audio_series = df["audio_array"]
        self.audios_list = self.audio_series.tolist()

        self.audio_feature_extractor = audio_feature_extractor
        self.max_length = 0
        self.audios = None

        if vectorise:
            self.extract_audio_features()
        else:
            # Get padded audio
            self.find_max_array_length()
            self.pad_audio()

        # Get tokenized text
        self.texts = [
            text_tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["sentence"]
        ]

    def extract_audio_features(self):
        """
        Extract audio features using preloaded feature extractor.
        :return: Assign a list of tensors to self.audios.
        """
        audios = []
        for audio in self.audio_series:
            # Extract the features
            extracted_tensor = self.audio_feature_extractor(
                audio,
                sampling_rate=16000,
                padding="max_length",
                truncation=True,
                max_length=200000,
                return_tensors="pt",
            )
            audios.append(extracted_tensor)
        # Reassign vectors as audios
        self.audios = audios

    def find_max_array_length(self):
        """
        Find maximum length of the inidividual audio arrays.
        :return: Assign an integer of maximum length to self.max_length.
        """
        list_len = [len(i) for i in self.audios_list]
        max_length = max(list_len)
        self.max_length = max_length

    def pad_audio(self):
        """
        Pad shorter audios with 0 to make all audio arrays equal lengths.
        :return: Assign a list of assrays to self.audios.
        """
        new_list = []
        for audio_array in self.audios_list:
            padded_audio_array = np.pad(
                audio_array, (0, self.max_length - len(audio_array)), "constant"
            )
            new_list.append(padded_audio_array)
        # audios is a list of arrays.
        self.audios = new_list

    def classes(self):
        """Get labels of each audio."""
        return self.labels

    def __len__(self):
        """Get the total number of audios in the dataset."""
        return len(self.labels)

    def get_batch_labels(self, idx):
        """Fetch a batch of labels."""
        return np.array(self.labels[idx])

    def get_batch_audios(self, idx):
        """Fetch a batch of inputs."""
        return self.audios[idx]

    def get_batch_texts(self, idx):
        """Fetch a batch of inputs."""
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_audios = self.get_batch_audios(idx)
        batch_texts = self.get_batch_texts(idx)
        # Put tensors to a list
        batch_audio_text = {"audio": batch_audios, "text": batch_texts}
        # batch_audio_text = [batch_audios, batch_texts]
        batch_y = self.get_batch_labels(idx)

        return batch_audio_text, batch_y


def train_audio_text(
    model,
    audio_feature_extractor,
    text_tokenizer,
    train_data,
    val_data,
    learning_rate,
    weight_decay,
    epochs,
    batch_size,
    num_workers,
    vectorise,
):
    """
    Train the model based on extracted audio vectors.
    :param model: Deep learning model for the audio training.
    :param audio_feature_extractor: Pre-trained transformer to extract audio features.
    :param text_tokenizer: Tokenizer for text.
    :param train_data: Dataframe to be trained.
    :param val_data: Dataframe to be evaluated.
    :param learning_rate: Parameter; rate of learning.
    :param weight_decay: Rate of decay; l2 regularisation.
    :param epochs: Number of epochs to be trained.
    :param batch_size: Number of batches.
    :param vectorise: Whether to vectorise audio.
    :return: Training and evaluation accuracies.
    """
    # Prepare data into dataloader
    train, val = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train, val = (
        AudioTextDataset(train, audio_feature_extractor, text_tokenizer, vectorise),
        AudioTextDataset(val, audio_feature_extractor, text_tokenizer, vectorise),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # # Freeze Bert/HuBert
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    #
    # for param in model.hubert.parameters():
    #     param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(tolerance=5, min_delta=1)
    writer = SummaryWriter()

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            # Audio
            input_values = train_input["audio"]["input_values"].squeeze(1).to(device)
            # Text
            mask = train_input["text"]["attention_mask"].to(device)
            input_id = train_input["text"]["input_ids"].squeeze(1).to(device)

            optimizer.zero_grad()
            output = model(input_values, input_id, mask)
            output = output.flatten()
            batch_loss = criterion(output.float(), train_label.float())
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, train_label)
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                # Audio
                input_values = val_input["audio"]["input_values"].squeeze(1).to(device)
                # Text
                mask = val_input["text"]["attention_mask"].to(device)
                input_id = val_input["text"]["input_ids"].squeeze(1).to(device)

                output = model(input_values, input_id, mask)
                output = output.flatten()
                batch_loss = criterion(output.float(), train_label.float())
                total_loss_val += batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                # Define accuracy as within 10% of the true label
                acc = test_accuracy(output, val_label)
                total_acc_val += acc

        # early stopping
        early_stopping(
            total_loss_train / len(train_data), total_loss_val / len(val_data)
        )
        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num)
            break

        # Add to tensorboard
        writer.add_scalar(
            "Training Loss", total_loss_train / len(train_data), epoch_num
        )
        writer.add_scalar(
            "Training Accuracy", total_acc_train / len(train_data), epoch_num
        )
        writer.add_scalar("Validation Loss", total_loss_val / len(val_data), epoch_num)
        writer.add_scalar(
            "Validation Accuracy", total_acc_val / len(val_data), epoch_num
        )
        writer.flush()
        # Append to list
        train_loss_list.append(total_loss_train / len(train_data))
        train_acc_list.append(total_acc_train / len(train_data))
        val_loss_list.append(total_loss_val / len(val_data))
        val_acc_list.append(total_acc_val / len(val_data))
        # Generate plots
        plot_name = "audio_text_"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        save_training_results(
            train_loss_list, train_acc_list, val_loss_list, val_acc_list, plot_name
        )

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

    # # Save plots and training results
    # gen_train_plots(train_loss_list, train_acc_list)
    # gen_eval_plots(val_loss_list, val_acc_list)
    # save_training_results(train_loss_list, train_acc_list, val_loss_list, val_acc_list)


# Save data to csv
def save_training_results(
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, plot_name
):
    """
    Save the results from model training.
    :param train_loss_list: List of training losses.
    :param train_acc_list: List of training accuracies.
    :param val_loss_list: List of evaluation losses.
    :param val_acc_list: List of evaluation accuracies.
    :param plot_name: Name of the plot depending on model.
    :return: Save results to a csv.
    """
    list_of_tuples = list(
        zip(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
    )
    training_results = pd.DataFrame(
        list_of_tuples, columns=["Train Loss", "Train Acc", "Eval Loss", "Eval Acc"]
    )
    training_results.to_csv(
        os.path.join("/home", "yyu", "plots", plot_name + "training_results.csv")
    )


def gen_acc_plots(train_acc_list, val_acc_list, plot_name):
    """
    Generate plots for training and evaluation accuracies.
    :param train_acc_list: List of training accuracies.
    :param val_acc_list: List of eval accuracies.
    :param plot_name: Name of the plot depending on model.
    :return: Save plot to directory.
    """
    plt.figure()
    epoch_list = list(range(len(train_acc_list)))
    plt.plot(epoch_list, train_acc_list, color="r", label="train")
    plt.plot(epoch_list, val_acc_list, color="b", label="val")
    plt.xlabel("Epoch Numbers")
    plt.ylabel("Accuracies")
    plt.title("Training and Evaluation Accuracies")
    plt.legend()
    save_path = os.path.join("/home", "yyu", "plots", plot_name + "acc.png")
    plt.savefig(save_path)
    # plt.show()


def gen_loss_plots(train_loss_list, val_loss_list, plot_name):
    """
    Generate plots for training and evaluation losses.
    :param train_loss_list: List of training losses.
    :param val_loss_list: List of eval losses.
    :param plot_name: Name of the plot depending on model.
    :return: Save plot to directory.
    """
    plt.figure()
    epoch_list = list(range(len(train_loss_list)))
    plt.plot(epoch_list, train_loss_list, color="r", label="train")
    plt.plot(epoch_list, val_loss_list, color="b", label="val")
    plt.xlabel("Epoch Numbers")
    plt.ylabel("losses")
    plt.title("Training and Evaluation losses")
    plt.legend()
    save_path = os.path.join("/home", "yyu", "plots", plot_name + "loss.png")
    plt.savefig(save_path)


def evaluate_audio_text(
    model, audio_feature_extractor, text_tokenizer, test_data, batch_size, vectorise
):
    """
    Evaluate accuracy for the model on vectorised audio data.
    :param model: Model to be used for deep learning.
    :param audio_feature_extractor: Pre-trained transformer to extract audio features.
    :param text_tokenizer: Tokenizer for text.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :return: Test Accuracies.
    """
    test = test_data.reset_index(drop=True)
    test = AudioTextDataset(test, audio_feature_extractor, text_tokenizer, vectorise)
    test_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, drop_last=True, pin_memory=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    total_acc_test = 0
    with torch.no_grad():
        model.eval()
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            # Audio
            input_values = test_input["audio"]["input_values"].squeeze(1).to(device)
            # Text
            mask = test_input["text"]["attention_mask"].to(device)
            input_id = test_input["text"]["input_ids"].squeeze(1).to(device)

            output = model(input_values, input_id, mask)
            output = output.flatten()

            # acc = (output.argmax(dim=1) == test_label).sum().item()
            # Define accuracy as within 10% of the true label
            acc = test_accuracy(output, test_label)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
