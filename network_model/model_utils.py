"""Utility functions for training and building models for text and audio
confidence classification.

----------------------------------
Class TextDataset: Class that handles the preparation of text for training.
Class AudioDataset: Class that handles the preparation of audio for training.
Class AudioTextDataset: Class that handles the preparation of both text and
                audio for training.
"""
import warnings

# import pingouin as pg
import nlpaug.augmenter.word as naw
import seaborn as sns
import wavio
import speech_recognition as sr
from scipy.io.wavfile import write
import nlpaug.augmenter.audio as naa
import os
import pandas as pd
import math
from prettytable import PrettyTable
import json
import numpy as np
import random
import gc
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label and title size
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20

num_gpus = torch.cuda.device_count()


def count_scores_in_bins(train_scores_list):
    """
    Count the scores in five bins.
    :param train_scores_list: List of all scores.
    :return: Five numbers of scores in each bin.
    """
    trains_scores_centered = [i - 2.5 for i in train_scores_list]
    first_bucket_count = 0
    second_bucket_count = 0
    third_bucket_count = 0
    fourth_bucket_count = 0
    fifth_bucket_count = 0
    for score in trains_scores_centered:
        if -2.5 <= score < -1.5:
            first_bucket_count += 1
        elif -1.5 <= score < -0.5:
            second_bucket_count += 1
        elif -0.5 <= score < 0.5:
            third_bucket_count += 1
        elif 0.5 <= score < 1.5:
            fourth_bucket_count += 1
        else:
            fifth_bucket_count += 1
    return (
        first_bucket_count,
        second_bucket_count,
        third_bucket_count,
        fourth_bucket_count,
        fifth_bucket_count,
    )


def plot_histogram_of_scores(home_dir, input_list, num_bins, plot_name, x_label):
    """
    Plot the histogram of scores.
    :param home_dir: Home directory.
    :param input_list: List of scores.
    :param num_bins: Number of bins.
    :param plot_name: Name of plot
    :param x_label: X label name.
    :return: Save histogram plot.
    """
    plt.figure()
    # plt.hist(input_list, bins=num_bins)
    # plt.xlabel(x_label)
    # plt.ylabel("Frequencies")
    # plt.title("Histogram of " + plot_name)
    # plt.savefig(os.path.join(home_dir, "plots", plot_name))
    # plt.show()
    # plt.clf()
    hist = sns.histplot(input_list, color="cornflowerblue", kde=True, bins=num_bins)
    plt.title("Histogram of " + plot_name, fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.savefig(os.path.join(home_dir, "plots", plot_name))


def split_to_train_val_test(home_dir):
    """
    Split the total crowdsourcing results to train, val and test
    :param home_dir:
    :return:
    """
    # Path for crowdsourcing results
    crowdsourcing_results_df_path = os.path.join(
        home_dir,
        "data_sheets",
        "crowdsourcing_results",
        "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned.csv",
    )

    crowdsourcing_results_train_df_path = os.path.join(
        home_dir,
        "data_sheets",
        "crowdsourcing_results",
        "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_train.csv",
    )

    crowdsourcing_results_val_df_path = os.path.join(
        home_dir,
        "data_sheets",
        "crowdsourcing_results",
        "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_val.csv",
    )

    crowdsourcing_results_test_df_path = os.path.join(
        home_dir,
        "data_sheets",
        "crowdsourcing_results",
        "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test.csv",
    )

    total_df = pd.read_csv(crowdsourcing_results_df_path)
    train_df = total_df[:3600]
    train_df.to_csv(crowdsourcing_results_train_df_path)
    val_df = total_df[3601:4050]
    val_df.to_csv(crowdsourcing_results_val_df_path)
    test_df = total_df[4051:]
    test_df.to_csv(crowdsourcing_results_test_df_path)


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
                max_length=120,
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


def test_accuracy(output, actual, absolute):
    """
    Testify whether the output is accurate.
    :param output: Score tensor output by model.
    :param actual: Actual score tensor.
    :param absolute: Whether to test with absolute value
    :return: Number of accurate predicitons
    """
    output_list = output.tolist()
    actual_list = actual.tolist()
    count = 0
    for i in range(len(output_list)):
        # If test by absolute value
        if absolute:
            if actual_list[i] - 0.5 <= output_list[i] <= actual_list[i] + 0.5:
                count += 1
        else:
            if actual_list[i] * 0.8 <= output_list[i] <= actual_list[i] * 1.2:
                count += 1
    return count


def calculate_mse(output_list, actual_list):
    """
    Calculate MSE between two lists.
    :param output_list: Score list output by model.
    :param actual_list: Actual score list.
    :return: MSE value.
    """
    mse = np.mean((np.array(actual_list) - np.array(output_list)) ** 2)
    return mse


def get_icc(output, actual, icc_type="ICC(3,1)"):
    """
    Get intraclass correlation score.
    :param output: Score list output by model.
    :param actual: Actual score list.
    :param icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k))
    :return: A floating number of ICC score.
    """
    Y = np.column_stack((output, actual))
    print(Y)
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()
    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc1":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == "ICC(2,1)" or icc_type == "ICC(2,k)":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        if icc_type == "ICC(2,k)":
            k = 1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == "ICC(3,1)" or icc_type == "ICC(3,k)":
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        if icc_type == "ICC(3,k)":
            k = 1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC


def calculate_mse(output_list, actual_list):
    """
    Calculate MSE between two lists.
    :param output_list: Score list output by model.
    :param actual_list: Actual score list.
    :return: MSE value.
    """
    mse = np.mean((actual_list - output_list) ** 2)
    return mse


# def get_icc(output, actual):
#     """
#     Get intraclass correlation score.
#     :param output: Score tensor output by model.
#     :param actual: Actual score tensor.
#     :return: A floating number of ICC score.
#     """
#     output_list = output.tolist()
#     actual_list = actual.tolist()
#
#     # create DataFrame
#     index_list = list(range(len(output_list)))
#     index_list_2 = index_list.copy()
#     index_list_2.extend(index_list)
#
#
#     icc_df = pd.DataFrame(
#         {
#             "index": index_list_2,
#             "rater": ["1"] * len(output_list) + ["2"] * len(actual_list),
#             "rating": [*output_list, *actual_list],
#         }
#     )
#     # icc_results_df = pg.intraclass_corr(
#     #     data=icc_df, targets="index", raters="rater", ratings="rating"
#     # )
#     # icc_value_df = icc_results_df.loc[icc_results_df.Type == "ICC3k", "ICC"]
#     # icc_value = icc_value_df.to_numpy()[0]
#     icc_value = 0
#     return icc_value


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
    test_absolute,
    accum_iter,
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
    :param num_workers: Number of workers.
    :param test_absolute: Whether to use absolute test.
    :param accum_iter: Number of epochs to accumulate before zero grad.
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
    train_output_list = []
    train_label_list = []
    val_loss_list = []
    val_acc_list = []
    val_output_list = []
    val_label_list = []

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            train_output = model(input_id, mask)
            train_output = train_output.flatten()
            train_output_list, train_label_list = append_to_list(
                train_output, train_label, train_output_list, train_label_list
            )
            batch_loss = criterion(train_output.float(), train_label.float())
            # normalize loss to account for batch accumulation
            batch_loss = batch_loss / accum_iter
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()

            acc = test_accuracy(train_output, train_label, test_absolute)
            total_acc_train += acc

            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)
                val_output = model(input_id, mask)
                val_output = val_output.flatten()
                val_output_list, val_label_list = append_to_list(
                    val_output, val_label, val_output_list, val_label_list
                )
                val_batch_loss = criterion(val_output.float(), val_label.float())
                # normalize loss to account for batch accumulation
                val_batch_loss = val_batch_loss / accum_iter
                total_loss_val += val_batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                acc = test_accuracy(val_output, val_label, test_absolute)
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
        plot_name = "text_simple_"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        save_training_results(
            train_loss_list,
            train_acc_list,
            train_output_list,
            train_label_list,
            val_loss_list,
            val_acc_list,
            val_output_list,
            val_label_list,
            plot_name,
        )

        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num)
            break

        # # Calculate icc values
        # train_icc = get_icc(train_output_list, train_label_list, icc_type="ICC(3,1)")
        # val_icc = get_icc(val_output_list, val_label_list, icc_type="ICC(3,1)")

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_text(model, test_data, tokenizer, batch_size, test_absolute):
    """
    Evaluate accuracy for the model on text data.
    :param model: Model to be used for deep learning.
    :param test_data: Dataframe to be tested.
    :param tokenizer: Pre-trained transformer to tokenize the text.
    :param batch_size: Number of batches.
    :param test_absolute: Whether to use absolute test.
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

            acc = test_accuracy(output, test_label, test_absolute)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


def augment_audio_random(audio):
    """
    Augment audio into new arrays.
    :param audio: Original audio in dataframe entries (list).
    :return: augmented audio array.
    """
    random_number = random.randint(0, 7)
    # print("length of input", len(audio))
    # If string, decode to list then apply array
    if isinstance(audio, str):
        audio = np.array(json.loads(audio))
    else:
        audio = np.array(audio)
    try:
        if random_number == 0:
            # Loudness
            aug = naa.LoudnessAug()
            augmented_data = aug.augment(audio)
        elif random_number == 1:
            # Noise
            aug = naa.NoiseAug()
            augmented_data = aug.augment(audio)
        elif random_number == 2:
            # Pitch
            aug = naa.PitchAug(sampling_rate=16000, factor=(2, 3))
            augmented_data = aug.augment(audio)
        elif random_number == 3:
            # Loudness and noise
            aug = naa.LoudnessAug()
            augmented_data = aug.augment(audio)
            aug = naa.NoiseAug()
            augmented_data = aug.augment(augmented_data)
        elif random_number == 4:
            # Loudness and pitch
            aug = naa.LoudnessAug()
            augmented_data = aug.augment(audio)
            aug = naa.PitchAug(sampling_rate=16000, factor=(2, 3))
            augmented_data = aug.augment(augmented_data)
        elif random_number == 5:
            # Noise and pitch
            aug = naa.NoiseAug()
            augmented_data = aug.augment(audio)
            aug = naa.PitchAug(sampling_rate=16000, factor=(2, 3))
            augmented_data = aug.augment(augmented_data)
        elif random_number == 6:
            # Loudness, noise and pitch
            aug = naa.LoudnessAug()
            augmented_data = aug.augment(audio)
            aug = naa.NoiseAug()
            augmented_data = aug.augment(augmented_data)
            aug = naa.PitchAug(sampling_rate=16000, factor=(2, 3))
            augmented_data = aug.augment(augmented_data)
        else:
            augmented_data = audio
    except Exception as e:
        print("Error!", e)
        # augmented_data = [audio, 2]
        augmented_data = audio
        pass

    # return augmented_data[0].tolist()
    return augmented_data


def augment_text_random(text):
    """
    Augment text into new string.
    :param text: Original text in dataframe entries (string).
    :return: augmented text.
    """
    random_number = random.randint(0, 5)
    try:
        if random_number == 0:
            # Contextual word embeddings
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased", action="insert"
            )
            augmented_text = aug.augment(text)[0]
        elif random_number == 1:
            # Substitute
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased", action="substitute"
            )
            augmented_text = aug.augment(text)[0]
        elif random_number == 2:
            # Delete word randomly
            aug = naw.RandomWordAug()
            augmented_text = aug.augment(text)[0]
        else:
            augmented_text = text
    except Exception as e:
        print("Error!", e)
        augmented_text = text
        pass

    return augmented_text


def augment_text_random_iter(sample_rate, result_df_augmented):
    """
    Augment text randomly.
    :param sample_rate: Sample rate for the audio.
    :param result_df_augmented: Original df with augmented audio.
    :return: Df with augmented audio and text.
    """
    # Find the feature csv locally
    for index, row in result_df_augmented.iterrows():
        random_number = random.randint(0, 6)
        if random_number == 0:
            # Speech to text translation
            wav_path = os.path.join("/home", "yyu", "wav_audio.wav")
            if isinstance(row["audio_array"], list):
                # Write the .wav file
                wavio.write(wav_path, row["audio_array"][0], sample_rate, sampwidth=2)
            elif isinstance(row["audio_array"], np.ndarray):
                wavio.write(wav_path, row["audio_array"], sample_rate, sampwidth=2)

            # initialize the recognizer
            r = sr.Recognizer()
            # open the file
            with sr.AudioFile(wav_path) as source:
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                try:
                    augmented_text = r.recognize_google(audio_data, language="en-IN")
                except:
                    # Use original text
                    augmented_text = row["sentence"]

            # Delete the wav file if exists
            if os.path.exists(wav_path):
                os.remove(wav_path)
            else:
                continue
        elif random_number == 1:
            # Contextual word embeddings
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased", action="insert"
            )
            augmented_text = aug.augment(row["sentence"])[0]
        elif random_number == 2:
            # Substitute
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased", action="substitute"
            )
            augmented_text = aug.augment(row["sentence"])[0]
        elif random_number == 3:
            # Delete word randomly
            aug = naw.RandomWordAug()
            augmented_text = aug.augment(row["sentence"])[0]
        else:
            augmented_text = row["sentence"]

        # Assign augmented text to original dataframe
        row["sentence"] = augmented_text

    return result_df_augmented


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
    home_dir,
    crowdsourcing_results_df_path,
    save_to_single_csv,
    augment_audio,
    two_scores,
):
    """
    Load the audio arrays and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :param augment_audio: Whether to augment the audio.
    :param:two_scores: Only use two scores from the worker.
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
            try:
                # Convert audio to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    # print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = [json.loads(i) for i in curr_audio_data]
                audio_list.append(curr_audio_data)

                if two_scores:
                    # Only take the most similar two answers
                    score = take_two_from_row(row)
                else:
                    score = row["average"] - 2.5
                score_list.append(score)

            except Exception as e:
                print("Error in parsing! File name = " + total_df_path)
                print(e)
                continue

    result_df = pd.DataFrame(
        np.column_stack([audio_list, score_list]), columns=["audio_array", "score"]
    )

    if augment_audio:
        result_df = upsample_and_augment_audio_only(result_df, times=1)
        print("size of final training dataset", result_df.shape[0])

    if save_to_single_csv:
        ## Save all data into a single csv file.
        save_path = os.path.join(
            home_dir, "data_sheets", "audio_only_crowd_all_model.csv"
        )
        result_df.to_csv(save_path, index=False)
    return result_df


def load_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_df_path,
    save_to_single_csv,
    augment_text,
    two_scores,
):
    """
    Load the text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :param augment_text: Whether to augment the text.
    :param two_scores: Whether to use only two scores
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
                # Convert to list
                curr_text_data = select_features_df["text"].to_list()[0]
                # print("curr text data", curr_text_data)
                # print(type(curr_text_data))
                text_list.append([curr_text_data])
                if two_scores:
                    # Only take the most similar two answers
                    score = take_two_from_row(row)
                else:
                    score = row["average"] - 2.5
                score_list.append(score)
            except Exception as e:
                print("Error in parsing! File name = " + all_features_csv_path)
                print(e)
                continue

    result_df = pd.DataFrame(
        np.column_stack([text_list, score_list]), columns=["sentence", "score"]
    )
    result_df["score"] = result_df["score"].astype(float)
    print("size of training dataset", result_df.shape[0])
    if augment_text:
        result_df = upsample_and_augment_text_only(result_df, times=1)
        print("size of final training dataset", result_df.shape[0])

    if save_to_single_csv:
        ## Save all data into a single csv file.
        save_path = os.path.join(
            home_dir, "data_sheets", "text_only_crowd_all_model_train.csv"
        )
        result_df.to_csv(save_path, index=False)
    return result_df


def upsample_audio_text_augment(
    upsample_train,
    augment_audio_train,
    augment_text_train,
    concat_final_train_df,
    home_dir,
    crowdsourcing_results_train_df_path,
    two_scores,
):
    """
    Upsample and augment the original training dataset.
    :param upsample_train: Whether to upsample the training dataset.
    :param augment_audio_train: Whether to augment audio.
    :param augment_text_train: Whether to augment text.
    :param concat_final_train_df: Concat the final dataframe.
    :param home_dir: Home directory.
    :param crowdsourcing_results_train_df_path: Path of the training df.
    :param two_scores: Whether to take two scores only.
    :return: Save created csvs.
    """
    if upsample_train:
        generate_train_data_from_crowdsourcing_results(
            home_dir,
            crowdsourcing_results_train_df_path,
            augment_audio=True,
            two_scores=two_scores,
        )
    if augment_audio_train:
        print("start augmenting audio!")
        for i in [1, 2, 3]:
            # Large dataset
            mylist = []
            for chunk in pd.read_csv(
                os.path.join(
                    "/home",
                    "yyu",
                    "data_sheets",
                    "audio_text_upsampled_unaugmented" + str(i) + ".csv",
                ),
                chunksize=1000,
            ):
                mylist.append(chunk)

            unaug_df = pd.concat(mylist, axis=0)
            del mylist

            augment_training_data_audio(unaug_df, number=i)
            print("finished the " + str(i) + "audio augmentation.")

    if augment_text_train:
        print("start augmenting text!")
        for i in [1, 2, 3]:
            # Large dataset
            mylistaudio = []
            for chunk in pd.read_csv(
                os.path.join(
                    "/home",
                    "yyu",
                    "data_sheets",
                    "audio_text_upsampled_audio_augmented" + str(i) + ".csv",
                ),
                chunksize=1000,
            ):
                mylistaudio.append(chunk)

            audio_aug = pd.concat(mylistaudio, axis=0)
            del mylistaudio

            augment_training_data_text(audio_aug, number=i)

    if concat_final_train_df:
        print("start concatenating final df!")
        total_df_list = []
        for i in [1, 2, 3]:
            # Individual dfs
            indiv_df_list = []
            for chunk in pd.read_csv(
                os.path.join(
                    "/home",
                    "yyu",
                    "data_sheets",
                    "audio_text_upsampled_audio_text_augmented" + str(i) + ".csv",
                ),
                chunksize=1000,
            ):
                indiv_df_list.append(chunk)

            audio_text_aug = pd.concat(indiv_df_list, axis=0)
            del indiv_df_list
            # Append to the large list
            total_df_list.append(audio_text_aug)

        # Concat to form the total dataframe.
        total_audio_text_aug = pd.concat(total_df_list)
        del total_df_list
        total_audio_text_aug.to_csv(
            os.path.join(
                "/home",
                "yyu",
                "data_sheets",
                "audio_text_upsampled_augmented_total.csv",
            )
        )


def upsample_training_data(result_df, times):
    """
    Upsample the dataframes in smaller buckets.
    :param result_df: Original unbalanced dataset.
    :param times: Number of times the total dataset size to be increased.
    :return: Balanced dataset dataframe.
    """
    print("start upsamping!")
    first_bucket_df = result_df.loc[result_df["score"] < -1.5]
    second_bucket_df = result_df.loc[
        (result_df["score"] >= -1.5) & (result_df["score"] < -0.5)
    ]
    third_bucket_df = result_df.loc[
        (result_df["score"] >= -0.5) & (result_df["score"] < 0.5)
    ]
    fourth_bucket_df = result_df.loc[
        (result_df["score"] >= 0.5) & (result_df["score"] < 1.5)
    ]
    fifth_bucket_df = result_df.loc[result_df["score"] >= 1.5]

    # Upsample the dfs
    num_rows_per_bucket = fourth_bucket_df.shape[0]
    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_first = num_rows_per_bucket / first_bucket_df.shape[0]
        first_bucket_df = first_bucket_df.sample(
            frac=num_repeat_first * times, replace=True, random_state=1
        )

    num_repeat_second = num_rows_per_bucket / second_bucket_df.shape[0]
    second_bucket_df = second_bucket_df.sample(
        frac=num_repeat_second * times, replace=True, random_state=1
    )

    num_repeat_third = num_rows_per_bucket / third_bucket_df.shape[0]
    third_bucket_df = third_bucket_df.sample(
        frac=num_repeat_third * times, replace=True, random_state=1
    )

    fourth_bucket_df = fourth_bucket_df.sample(frac=times, replace=True, random_state=1)

    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_fifth = num_rows_per_bucket / fifth_bucket_df.shape[0]
        fifth_bucket_df = fifth_bucket_df.sample(
            frac=num_repeat_fifth * times, replace=True, random_state=1
        )
    all_dfs = [
        first_bucket_df,
        second_bucket_df,
        third_bucket_df,
        fourth_bucket_df,
        fifth_bucket_df,
    ]
    result_df = pd.concat(all_dfs)
    # result_df.to_csv(
    #     os.path.join(
    #         "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented.csv",
    #     )
    # )

    # Get total number of rows
    n_rows = result_df.shape[0]
    print("size of result df", n_rows)
    chunk_size = math.floor(n_rows / 3)

    # Total 18720 rows
    unaug_df1 = result_df[:chunk_size]
    unaug_df1.to_csv(
        os.path.join(
            "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented1.csv",
        )
    )

    unaug_df2 = result_df[chunk_size + 1 : chunk_size * 2]
    unaug_df2.to_csv(
        os.path.join(
            "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented2.csv",
        )
    )

    unaug_df3 = result_df[chunk_size * 2 :]
    unaug_df3.to_csv(
        os.path.join(
            "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented3.csv",
        )
    )

    return result_df


def augment_training_data_audio(upsampled_df, number):
    """
    Augment the upsampled dataframe and save to csv.
    :param upsampled_df: Training dataframe that has been upsampled.
    :param number: The number of the unaugmented sub dataframe.
    :return:
    """
    print("Start augmenting!")
    # Augment audio
    upsampled_df["audio_array"] = upsampled_df["audio_array"].apply(
        augment_audio_random
    )

    upsampled_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "data_sheets",
            "audio_text_upsampled_audio_augmented" + str(number) + ".csv",
        )
    )


def augment_training_data_text(audio_aug_df, number):
    """
    Augment text to complete the training dataset and save to csv.
    :param audio_aug_df: Dataframe upsampled and augmented with audio.
    :param number: The number of the unaugmented sub dataframe.
    :return:
    """
    # Augment text
    print("augmenting text!")
    sample_rate = 22050
    result_df_augmented_audio_text = augment_text_random_iter(
        sample_rate=sample_rate, result_df_augmented=audio_aug_df
    )

    result_df_augmented_audio_text.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "data_sheets",
            "audio_text_upsampled_audio_text_augmented" + str(number) + ".csv",
        )
    )


def upsample_and_augment_audio_only(result_df, times):
    """
    Upsample the dataframes in smaller buckets and augment audio data.
    :param result_df: Original unbalanced dataset.
    :param times: Number of times the total dataset size to be increased.
    :return: Balanced dataset dataframe.
    """
    print("start upsamping!")
    first_bucket_df = result_df.loc[result_df["score"] < -1.5]
    second_bucket_df = result_df.loc[
        (result_df["score"] >= -1.5) & (result_df["score"] < -0.5)
    ]
    third_bucket_df = result_df.loc[
        (result_df["score"] >= -0.5) & (result_df["score"] < 0.5)
    ]
    fourth_bucket_df = result_df.loc[
        (result_df["score"] >= 0.5) & (result_df["score"] < 1.5)
    ]
    fifth_bucket_df = result_df.loc[result_df["score"] >= 1.5]

    # Upsample the dfs
    num_rows_per_bucket = fourth_bucket_df.shape[0]
    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_first = num_rows_per_bucket / first_bucket_df.shape[0]
        first_bucket_df = first_bucket_df.sample(
            frac=num_repeat_first * times, replace=True, random_state=1
        )

    num_repeat_second = num_rows_per_bucket / second_bucket_df.shape[0]
    second_bucket_df = second_bucket_df.sample(
        frac=num_repeat_second * times, replace=True, random_state=1
    )

    num_repeat_third = num_rows_per_bucket / third_bucket_df.shape[0]
    third_bucket_df = third_bucket_df.sample(
        frac=num_repeat_third * times, replace=True, random_state=1
    )

    fourth_bucket_df = fourth_bucket_df.sample(frac=times, replace=True, random_state=1)

    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_fifth = num_rows_per_bucket / fifth_bucket_df.shape[0]
        fifth_bucket_df = fifth_bucket_df.sample(
            frac=num_repeat_fifth * times, replace=True, random_state=1
        )
    all_dfs = [
        first_bucket_df,
        second_bucket_df,
        third_bucket_df,
        fourth_bucket_df,
        fifth_bucket_df,
    ]
    result_df = pd.concat(all_dfs)

    # result_df.to_csv(
    #     os.path.join(
    #         "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented.csv",
    #     )
    # )
    # Delete individual dataframes
    del first_bucket_df
    del second_bucket_df
    del third_bucket_df
    del fourth_bucket_df
    del fifth_bucket_df
    del all_dfs
    gc.collect()
    first_bucket_df = pd.DataFrame()
    second_bucket_df = pd.DataFrame()
    third_bucket_df = pd.DataFrame()
    fourth_bucket_df = pd.DataFrame()
    fifth_bucket_df = pd.DataFrame()
    print("Deleted all individual dfs")

    # Augment audio
    result_df["audio_array"] = result_df["audio_array"].apply(augment_audio_random)

    return result_df


def upsample_and_augment_text_only(result_df, times):
    """
    Upsample the dataframes in smaller buckets and augment text data.
    :param result_df: Original unbalanced dataset.
    :param times: Number of times the total dataset size to be increased.
    :return: Balanced dataset dataframe.
    """
    print("start upsamping!")
    first_bucket_df = result_df.loc[result_df["score"] < -1.5]
    second_bucket_df = result_df.loc[
        (result_df["score"] >= -1.5) & (result_df["score"] < -0.5)
    ]
    third_bucket_df = result_df.loc[
        (result_df["score"] >= -0.5) & (result_df["score"] < 0.5)
    ]
    fourth_bucket_df = result_df.loc[
        (result_df["score"] >= 0.5) & (result_df["score"] < 1.5)
    ]
    fifth_bucket_df = result_df.loc[result_df["score"] >= 1.5]

    # Upsample the dfs
    num_rows_per_bucket = fourth_bucket_df.shape[0]
    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_first = num_rows_per_bucket / first_bucket_df.shape[0]
        first_bucket_df = first_bucket_df.sample(
            frac=num_repeat_first * times, replace=True, random_state=1
        )

    num_repeat_second = num_rows_per_bucket / second_bucket_df.shape[0]
    second_bucket_df = second_bucket_df.sample(
        frac=num_repeat_second * times, replace=True, random_state=1
    )

    num_repeat_third = num_rows_per_bucket / third_bucket_df.shape[0]
    third_bucket_df = third_bucket_df.sample(
        frac=num_repeat_third * times, replace=True, random_state=1
    )

    fourth_bucket_df = fourth_bucket_df.sample(frac=times, replace=True, random_state=1)

    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_fifth = num_rows_per_bucket / fifth_bucket_df.shape[0]
        fifth_bucket_df = fifth_bucket_df.sample(
            frac=num_repeat_fifth * times, replace=True, random_state=1
        )
    all_dfs = [
        first_bucket_df,
        second_bucket_df,
        third_bucket_df,
        fourth_bucket_df,
        fifth_bucket_df,
    ]
    result_df = pd.concat(all_dfs)

    # result_df.to_csv(
    #     os.path.join(
    #         "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented.csv",
    #     )
    # )
    # Delete individual dataframes
    del first_bucket_df
    del second_bucket_df
    del third_bucket_df
    del fourth_bucket_df
    del fifth_bucket_df
    del all_dfs
    gc.collect()
    first_bucket_df = pd.DataFrame()
    second_bucket_df = pd.DataFrame()
    third_bucket_df = pd.DataFrame()
    fourth_bucket_df = pd.DataFrame()
    fifth_bucket_df = pd.DataFrame()
    print("Deleted all individual dfs")

    # Augment audio
    result_df["sentence"] = result_df["sentence"].apply(augment_text_random)

    return result_df


def upsample_and_augment(result_df, times):
    """
    Upsample the dataframes in smaller buckets and augment audio data.
    :param result_df: Original unbalanced dataset.
    :param times: Number of times the total dataset size to be increased.
    :return: Balanced dataset dataframe.
    """
    print("start upsamping!")
    first_bucket_df = result_df.loc[result_df["score"] < -1.5]
    second_bucket_df = result_df.loc[
        (result_df["score"] >= -1.5) & (result_df["score"] < -0.5)
    ]
    third_bucket_df = result_df.loc[
        (result_df["score"] >= -0.5) & (result_df["score"] < 0.5)
    ]
    fourth_bucket_df = result_df.loc[
        (result_df["score"] >= 0.5) & (result_df["score"] < 1.5)
    ]
    fifth_bucket_df = result_df.loc[result_df["score"] >= 1.5]

    # Upsample the dfs
    num_rows_per_bucket = fourth_bucket_df.shape[0]
    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_first = num_rows_per_bucket / first_bucket_df.shape[0]
        first_bucket_df = first_bucket_df.sample(
            frac=num_repeat_first * times, replace=True, random_state=1
        )

    num_repeat_second = num_rows_per_bucket / second_bucket_df.shape[0]
    second_bucket_df = second_bucket_df.sample(
        frac=num_repeat_second * times, replace=True, random_state=1
    )

    num_repeat_third = num_rows_per_bucket / third_bucket_df.shape[0]
    third_bucket_df = third_bucket_df.sample(
        frac=num_repeat_third * times, replace=True, random_state=1
    )

    fourth_bucket_df = fourth_bucket_df.sample(frac=times, replace=True, random_state=1)

    if first_bucket_df.shape[0] == 0:
        pass
    else:
        num_repeat_fifth = num_rows_per_bucket / fifth_bucket_df.shape[0]
        fifth_bucket_df = fifth_bucket_df.sample(
            frac=num_repeat_fifth * times, replace=True, random_state=1
        )
    all_dfs = [
        first_bucket_df,
        second_bucket_df,
        third_bucket_df,
        fourth_bucket_df,
        fifth_bucket_df,
    ]
    result_df = pd.concat(all_dfs)

    # result_df.to_csv(
    #     os.path.join(
    #         "/home", "yyu", "data_sheets", "audio_text_upsampled_unaugmented.csv",
    #     )
    # )
    # Delete individual dataframes
    del first_bucket_df
    del second_bucket_df
    del third_bucket_df
    del fourth_bucket_df
    del fifth_bucket_df
    del all_dfs
    gc.collect()
    first_bucket_df = pd.DataFrame()
    second_bucket_df = pd.DataFrame()
    third_bucket_df = pd.DataFrame()
    fourth_bucket_df = pd.DataFrame()
    fifth_bucket_df = pd.DataFrame()
    print("Deleted all individual dfs")

    # # Augment audio
    # result_df["audio_array"] = result_df["audio_array"].apply(augment_audio_random)
    #
    # # # Augment text
    # sample_rate = 16000
    # result_df_augmented_audio_text = augment_text_random_iter(
    #     sample_rate=sample_rate, result_df_augmented=result_df
    # )
    #
    # # Delete dataframes and list to free memory
    # lst = [result_df]
    # del result_df
    # del lst
    # gc.collect()
    # result_df = pd.DataFrame()
    # print("Deleted result_df")

    return result_df


def take_two_from_row(row):
    """
    Select the two most similar scores from the row and take average, then shift by 2.5.
    :param row: Pandas dataframe row
    :return: An average score.
    """
    # Take two scores that agree better with each other
    diff_one_two = abs(row["score1"] - row["score2"])
    diff_two_three = abs(row["score2"] - row["score3"])
    diff_one_three = abs(row["score1"] - row["score3"])
    diff_list = [diff_one_two, diff_two_three, diff_one_three]
    val, idx = min((val, idx) for (idx, val) in enumerate(diff_list))
    if idx == 0:
        score = (row["score1"] + row["score2"]) / 2 - 2.5
    elif idx == 1:
        score = (row["score2"] + row["score3"]) / 2 - 2.5
    else:
        score = (row["score1"] + row["score3"]) / 2 - 2.5
    return score


def load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_df_path,
    save_to_single_csv,
    augment_audio,
    two_scores,
):
    """
    Load the audio arrays, text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param save_to_single_csv: Whether to save to a single csv file.
    :param augment_audio: Whether to augment audio.
    :param two_scores: Whether to only use two scores.
    :return: Dataframe of audio arrays, text and average score.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    # Initialise empty lists
    audio_path_list = []
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
        audio_path = os.path.join(
            home_dir, "extracted_audios", str(folder_number), segment_name + ".mp3",
        )
        # Only proceed if file exists
        if os.path.isfile(total_df_path):
            total_df = pd.read_csv(total_df_path, encoding="utf-8", dtype="unicode")
            try:
                # Convert audio to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    # print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = [json.loads(i) for i in curr_audio_data]
                audio_path_list.append(audio_path)
                audio_list.append(curr_audio_data)

                # Convert text to list
                curr_text_data = total_df["text"].to_list()[0]
                text_list.append([curr_text_data])
                if two_scores:
                    # Only take the most similar two answers
                    score = take_two_from_row(row)
                else:
                    score = row["average"] - 2.5
                score_list.append(score)

            except Exception as e:
                print("Error in parsing! File name = " + total_df_path)
                print(e)
                continue

    result_df = pd.DataFrame(
        np.column_stack([audio_path_list, audio_list, text_list, score_list]),
        columns=["audio_path", "audio_array", "sentence", "score"],
    )

    if augment_audio:
        result_df = upsample_and_augment(result_df, times=1)
    print("size of final training dataset", result_df.shape[0])
    if save_to_single_csv:
        if augment_audio:
            ## Save all data into a single csv file.
            save_path = os.path.join(
                home_dir, "data_sheets", "audio_text_crowd_all_model_upsample.csv"
            )
        else:
            ## Save all data into a single csv file.
            save_path = os.path.join(
                home_dir, "data_sheets", "audio_text_crowd_test.csv"
            )
        result_df.to_csv(save_path, index=False)
    return result_df


def generate_train_data_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, augment_audio, two_scores,
):
    """
    Load the audio arrays, text and user scores from the csv files.
    :param home_dir: Home directory.
    :param crowdsourcing_results_df_path: Path to the results dataframe.
    :param augment_audio: Whether to augment audio.
    :param two_scores: Whether to only use two scores.
    :return: Dataframe of audio arrays, text and average score.
    """
    # Load crowdsourcing results df
    results_df = pd.read_csv(crowdsourcing_results_df_path)
    # Initialise empty lists
    audio_path_list = []
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
        audio_path = os.path.join(
            home_dir, "extracted_audios", str(folder_number), segment_name + ".mp3",
        )
        # Only proceed if file exists
        if os.path.isfile(total_df_path):
            total_df = pd.read_csv(total_df_path, encoding="utf-8", dtype="unicode")
            try:
                # Convert audio to list
                curr_audio_data = total_df["audio_array"].to_list()
                # If list contains element of type string
                if not all(isinstance(i, float) for i in curr_audio_data):
                    # print("Found wrong data type!")
                    # Decode to float using json
                    curr_audio_data = [json.loads(i) for i in curr_audio_data]
                audio_path_list.append(audio_path)
                audio_list.append(curr_audio_data)

                # Convert text to list
                curr_text_data = total_df["text"].to_list()[0]
                text_list.append([curr_text_data])
                if two_scores:
                    # Only take the most similar two answers
                    score = take_two_from_row(row)
                else:
                    score = row["average"] - 2.5
                score_list.append(score)

            except Exception as e:
                print("Error in parsing! File name = " + total_df_path)
                print(e)
                continue

    result_df = pd.DataFrame(
        np.column_stack([audio_path_list, audio_list, text_list, score_list]),
        columns=["audio_path", "audio_array", "sentence", "score"],
    )

    if augment_audio:
        upsampled_df = upsample_training_data(result_df, times=3)
        print("size of final training dataset", upsampled_df.shape[0])
        # augment_training_data(upsampled_df)


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
    test_absolute,
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
    :param test_absolute: Whether to use absolute test.
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

            acc = test_accuracy(output, train_label, test_absolute)
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

                acc = test_accuracy(output, val_label, test_absolute)
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


def evaluate_select_features(
    test_data, batch_size, num_of_rows, num_of_columns, test_absolute
):
    """
    Evaluate accuracy for the model on text data.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param num_of_rows: Maximum row number in the whole dataset.
    :param num_of_columns: Number of columns in final df.
    :param test_absolute: Whether to use absolute test.
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

            acc = test_accuracy(output, test_label, test_absolute)
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
                max_length=300000,
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
    test_absolute,
    accum_iter,
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
    :param test_absolute: Whether to use absolute test.
    :param accum_iter: Number of epochs to iterate until no grad.
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
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    train_loss_list = []
    train_acc_list = []
    train_output_list = []
    train_label_list = []
    val_loss_list = []
    val_acc_list = []
    val_output_list = []
    val_label_list = []

    for param in model.hubert.parameters():
        param.requires_grad = False

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

            train_output = model(input_values)
            train_output = train_output.flatten()
            train_output_list, train_label_list = append_to_list(
                train_output, train_label, train_output_list, train_label_list
            )
            batch_loss = criterion(train_output.float(), train_label.float())
            # normalize loss to account for batch accumulation
            batch_loss = batch_loss / accum_iter
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()

            acc = test_accuracy(train_output, train_label, test_absolute)
            total_acc_train += acc

            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                # mask = val_input["attention_mask"].to(device)
                if vectorise:
                    input_values = val_input["input_values"].squeeze(1).to(device)
                else:
                    # print("train input type", type(train_input))
                    # print(train_input)
                    # print(type(train_input["input_values"]))
                    # input_values = train_input["input_values"].to(device, dtype=torch.float)
                    # print(input_values.size())
                    input_values = val_input.to(device, dtype=torch.float)

                val_output = model(input_values)
                val_output = val_output.flatten()
                val_output_list, val_label_list = append_to_list(
                    val_output, val_label, val_output_list, val_label_list
                )

                val_batch_loss = criterion(val_output.float(), val_label.float())
                # normalize loss to account for batch accumulation
                val_batch_loss = val_batch_loss / accum_iter
                total_loss_val += val_batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()
                acc = test_accuracy(val_output, val_label, test_absolute)
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
        plot_name = "audio_simple_"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        save_training_results(
            train_loss_list,
            train_acc_list,
            train_output_list,
            train_label_list,
            val_loss_list,
            val_acc_list,
            val_output_list,
            val_label_list,
            plot_name,
        )

        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num)
            break

        # # Calculate icc values
        # train_icc = get_icc(train_output_list, train_label_list, icc_type="ICC(3,1)")
        # val_icc = get_icc(val_output_list, val_label_list, icc_type="ICC(3,1)")

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                            | Val Loss: {total_loss_val / len(val_data): .3f} \
                            | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate_audio(
    model, test_data, batch_size, feature_extractor, vectorise, test_absolute
):
    """
    Evaluate accuracy for the model on vectorised audio data.
    :param model: Model to be used for deep learning.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param feature_extractor: Pre-trained transformer to extract audio features.
    :param vectorise: If vectorised, use transformers to tokenize audios.
    :param test_absolute: Whether to use absolute test.
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

            acc = test_accuracy(output, test_label, test_absolute)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


class AudioTextDataset(torch.utils.data.Dataset):
    """
    Vectorise the audio arrays and text using the transformers and prepare
    as dataset.
    """

    def __init__(self, df, audio_feature_extractor, text_tokenizer, vectorise):
        self.df = df
        self.labels = df["score"].tolist()

        # Get audio
        self.audio_series = df["audio_array"]
        # self.audios_list = self.audio_series.tolist()

        self.audio_feature_extractor = audio_feature_extractor
        self.max_length = 0
        self.audios = None

        if vectorise:
            self.extract_audio_features()
        else:
            # Get padded audio
            # If string
            if isinstance(self.audio_series[0], str):
                self.audios_list = [
                    json.loads(element) for element in self.audio_series
                ]
            # If not string
            else:
                self.audios_list = self.audio_series.tolist()
            self.find_max_array_length()
            self.pad_audio()

        # Get tokenized text
        self.texts = [
            text_tokenizer(
                text,
                padding="max_length",
                max_length=120,
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
            # if isinstance(audio, str):
            #     audio = json.loads(audio)
            # else:
            #     pass
            # Extract the features
            extracted_tensor = self.audio_feature_extractor(
                audio,
                sampling_rate=16000,
                padding="max_length",
                truncation=True,
                max_length=300000,
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
    accum_iter,
    vectorise,
    test_absolute,
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
    :param accum_iter: Number of batches to be iterated before optimizer step.
    :param vectorise: Whether to vectorise audio.
    :param test_absolute: Whether to use absolute test.
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

    # Freezing selected model parameters
    for name, param in list(model.bert.named_parameters())[:195]:
        param.requires_grad = False
    for name, param in list(model.hubert.named_parameters())[:200]:
        param.requires_grad = False

    # # Freeze Bert/HuBert
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    #
    # for param in model.hubert.parameters():
    #     param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

    if use_cuda:
        print("Using cuda!")
        model = model.to(device)
        # count_parameters(model)
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        criterion = criterion.cuda()

    train_loss_list = []
    train_acc_list = []
    train_output_list = []
    train_label_list = []
    val_loss_list = []
    val_acc_list = []
    val_output_list = []
    val_label_list = []

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

            with torch.cuda.amp.autocast():
                train_output = model(input_values, input_id, mask)
                train_output = train_output.flatten()
                train_output_list, train_label_list = append_to_list(
                    train_output.cpu(),
                    train_label.cpu(),
                    train_output_list,
                    train_label_list,
                )
                batch_loss = criterion(train_output.float(), train_label.float())
                # normalize loss to account for batch accumulation
                batch_loss = batch_loss / accum_iter
                total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            acc = test_accuracy(train_output, train_label, test_absolute)
            total_acc_train += acc
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                # Audio
                val_input_values = (
                    val_input["audio"]["input_values"].squeeze(1).to(device)
                )
                # Text
                val_mask = val_input["text"]["attention_mask"].to(device)
                val_input_id = val_input["text"]["input_ids"].squeeze(1).to(device)

                val_output = model(val_input_values, val_input_id, val_mask)
                val_output = val_output.flatten()
                # Append results to the val lists
                val_output_list, val_label_list = append_to_list(
                    val_output.cpu(), val_label.cpu(), val_output_list, val_label_list
                )


                val_batch_loss = criterion(val_output.float(), val_label.float())
                # normalize loss to account for batch accumulation
                val_batch_loss = val_batch_loss / accum_iter
                total_loss_val += val_batch_loss.item()

                # acc = (output.argmax(dim=1) == val_label).sum().item()

                val_acc = test_accuracy(val_output, val_label, test_absolute)
                total_acc_val += val_acc

        # early stopping
        early_stopping(
            total_loss_train / len(train_data), total_loss_val / len(val_data)
        )
        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num + 1)
            break

        # Append to list
        train_loss_list.append(total_loss_train / len(train_data))
        train_acc_list.append(total_acc_train / len(train_data))
        val_loss_list.append(total_loss_val / len(val_data))
        val_acc_list.append(total_acc_val / len(val_data))

        # Generate plots
        plot_name = "multi_upsample_three_augment_audio_unfreeze_5-7"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        gen_val_scatter_plot(val_output_list, val_label_list, plot_name)
        save_training_results(
            train_loss_list,
            train_acc_list,
            train_output_list,
            train_label_list,
            val_loss_list,
            val_acc_list,
            val_output_list,
            val_label_list,
            plot_name,
        )
        # # Calculate icc values
        # train_icc = get_icc(train_output_list, train_label_list, icc_type="ICC(3,1)")
        # val_icc = get_icc(val_output_list, val_label_list, icc_type="ICC(3,1)")

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

        gc.collect()
        torch.cuda.empty_cache()



def append_to_list(output, label, output_list, label_list):
    """
    Append output score and label scores into the lists.
    :param output: Output tensor from the model.
    :param label: Label tensor.
    :param output_list: List of outputs for validation.
    :param label_list: List of true label for validation.
    :return:
    """
    for i in output.tolist():
        output_list.append(i)
    for l in label.tolist():
        label_list.append(l)

    return output_list, label_list


# Save data to csv
def save_training_results(
    train_loss_list,
    train_acc_list,
    train_output_list,
    train_label_list,
    val_loss_list,
    val_acc_list,
    val_output_list,
    val_label_list,
    plot_name,
):
    """
    Save the results from model training.
    :param train_loss_list: List of training losses.
    :param train_acc_list: List of training accuracies.
    :param train_output_list: List of training output.
    :param train_label_list: List of tensors of training labels.
    :param val_loss_list: List of evaluation losses.
    :param val_acc_list: List of evaluation accuracies.
    :param val_output_list: List of val output.
    :param val_label_list: List of tensors of val labels.
    :param plot_name: Name of the plot depending on model.
    :return: Save results to a csv.
    """
    list_of_tuples_loss_acc = list(
        zip(train_loss_list, train_acc_list, val_loss_list, val_acc_list,)
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_loss_acc,
        columns=["Train Loss", "Train Acc", "Val Loss", "Val Acc",],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home", "yyu", "plots", "training_csv", plot_name + "loss_acc.csv"
        ),
        index=False,
    )

    list_of_tuples_output = list(
        zip(train_output_list, train_label_list, val_output_list, val_label_list,)
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_output,
        columns=["Train Output", "Train Label", "Val Output", "Val Label",],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home", "yyu", "plots", "training_csv", plot_name + "output_label.csv"
        ),
        index=False,
    )

    # my_data = {"Train Loss": train_loss_list, "Train Acc": train_acc_list, "Train Output": train_output_list,
    #            "Train Label": train_label_list, "Val Loss": val_loss_list, "Val Acc": val_acc_list, "Val Output": val_output_list,
    #            "Val Label": val_label_list}
    #
    # my_data_dict = dict([(k, pd.Series(v)) for k, v in my_data.items()])
    # my_df = pd.DataFrame(my_data_dict)
    # my_df.to_csv(
    #     os.path.join("/home", "yyu", "plots", plot_name + "training_results.csv")
    # )


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
    plt.clf()
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
    plt.clf()


def gen_val_scatter_plot(val_output_list, val_label_list, plot_name):
    plt.figure()
    plt.scatter(val_label_list, val_output_list)
    plt.xlabel("Ground Truth Scores")
    plt.ylabel("Model Output Scores")
    plt.title("Model output and ground truth for validation")
    save_path = os.path.join("/home", "yyu", "plots", plot_name + "val_scatter.png")
    plt.savefig(save_path)
    plt.clf()


def evaluate_audio_text(
    model,
    audio_feature_extractor,
    text_tokenizer,
    test_data,
    batch_size,
    vectorise,
    test_absolute,
):
    """
    Evaluate accuracy for the model on vectorised audio data.
    :param model: Model to be used for deep learning.
    :param audio_feature_extractor: Pre-trained transformer to extract audio features.
    :param text_tokenizer: Tokenizer for text.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param test_absolute: Whether to use absolute test.
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

            acc = test_accuracy(output, test_label, test_absolute)
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
