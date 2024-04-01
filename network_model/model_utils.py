"""Utility functions for training and building models for text and audio
confidence classification.

----------------------------------
Class TextDataset: Class that handles the preparation of text for training.
Class AudioDataset: Class that handles the preparation of audio for training.
Class AudioTextDataset: Class that handles the preparation of both text and
                audio for training.
"""

import warnings

import pingouin as pg
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
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from models import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label and title size
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20

num_gpus = torch.cuda.device_count()


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
    hist = sns.histplot(input_list, color="cornflowerblue", kde=True, bins=num_bins)
    plt.title("Histogram of " + plot_name, fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.savefig(os.path.join(home_dir, "plots", plot_name))


def split_to_train_val(home_dir):
    """
    Split the total crowdsourcing results to train, val and test
    :param home_dir:
    :return:
    """
    # Path for crowdsourcing results
    crowdsourcing_results_df_path = os.path.join(
        home_dir,
        "data",
        "label_results",
        "Cleaned_Results_Removed.csv",
    )

    crowdsourcing_results_train_df_path = os.path.join(
        home_dir,
        "label_results",
        "Cleaned_Results_Train.csv",
    )

    crowdsourcing_results_val_df_path = os.path.join(
        home_dir,
        "data_sheets",
        "label_results",
        "Cleaned_Results_Val.csv",
    )

    # Test df is fixed to match the human labels; test df corresponds to the total_df[4051:]. Hence we only extract the first 4050 from total df to form train and val datasets.
    total_df = pd.read_csv(crowdsourcing_results_df_path)[:4050]
    total_df = total_df.sample(frac=1).reset_index(drop=True)
    train_df = total_df[:3600]
    train_df.to_csv(crowdsourcing_results_train_df_path)
    val_df = total_df[3601:]
    val_df.to_csv(crowdsourcing_results_val_df_path)


def get_accuracy(output, actual, absolute):
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


def get_icc(output, actual, icc_type="ICC(3,1)"):
    """
    Get intraclass correlation score.
    :param output: Score list output by model.
    :param actual: Actual score list.
    :param icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k))
    :return: A floating number of ICC score.
    """
    Y = np.column_stack((output, actual))
    # print(Y)
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
    SSE = (residuals**2).sum()
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


def get_mse(output_list, actual_list):
    """
    Calculate MSE between two lists.
    :param output_list: Score list output by model.
    :param actual_list: Actual score list.
    :return: MSE value.
    """
    mse = np.mean((actual_list - output_list) ** 2)
    return mse


def get_icc(output, actual):
    """
    Get intraclass correlation score.
    :param output: Score tensor output by model.
    :param actual: Actual score tensor.
    :return: A floating number of ICC score.
    """
    output_list = output.tolist()
    actual_list = actual.tolist()

    # create DataFrame
    index_list = list(range(len(output_list)))
    index_list_2 = index_list.copy()
    index_list_2.extend(index_list)

    icc_df = pd.DataFrame(
        {
            "index": index_list_2,
            "rater": ["1"] * len(output_list) + ["2"] * len(actual_list),
            "rating": [*output_list, *actual_list],
        }
    )
    icc_results_df = pg.intraclass_corr(
        data=icc_df, targets="index", raters="rater", ratings="rating"
    )
    icc_value_df = icc_results_df.loc[icc_results_df.Type == "ICC3k", "ICC"]
    icc_value = icc_value_df.to_numpy()[0]
    return icc_value


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
        else:
            augmented_data = audio
    except Exception as e:
        print("Error!", e)
        # augmented_data = [audio, 2]
        augmented_data = audio
        pass

    # return augmented_data[0].tolist()
    return augmented_data


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

    # Get total number of rows
    n_rows = result_df.shape[0]
    print("size of result df", n_rows)
    chunk_size = math.floor(n_rows / 3)

    # Total 18720 rows
    unaug_df1 = result_df[:chunk_size]
    unaug_df1.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "data_sheets",
            "audio_text_upsampled_unaugmented1.csv",
        )
    )

    unaug_df2 = result_df[chunk_size + 1 : chunk_size * 2]
    unaug_df2.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "data_sheets",
            "audio_text_upsampled_unaugmented2.csv",
        )
    )

    unaug_df3 = result_df[chunk_size * 2 :]
    unaug_df3.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "data_sheets",
            "audio_text_upsampled_unaugmented3.csv",
        )
    )

    return result_df


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

    return result_df


def upsample_and_augment_audio(result_df, times):
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
            home_dir,
            "extracted_audios",
            str(folder_number),
            segment_name + ".mp3",
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
        result_df = upsample_and_augment_audio(result_df, times=1)
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


def train_audio_text_handler_model(label, input, device, model, criterion):
    """
    Handler for training and val steps.
    :param label: Label.
    :param input: Inputs.
    :param device: CUDA or CPU.
    :param model: Model to be trained.
    :param criterion: Criterion to use.
    :return: acc, batch_loss, output, label.
    """
    label = label.to(device)
    # Audio
    input_values = input["audio"]["input_values"].squeeze(1).to(device)
    # Text
    mask = input["text"]["attention_mask"].to(device)
    input_id = input["text"]["input_ids"].squeeze(1).to(device)

    output = model(input_values, input_id, mask)
    output = output.flatten()
    batch_loss = criterion(output.float(), label.float())

    # acc = (output.argmax(dim=1) == train_label).sum().item()

    acc = get_accuracy(output, label, absolute=True)

    return acc, batch_loss, output, label


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
    freeze,
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

    # Rules for freezing
    if freeze == "first_ten":
        for layer in model.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in model.hubert.encoder.layers[:11]:
            for param in layer.parameters():
                param.requires_grad = False

    # Rules for freezing
    if freeze == "first_ele":
        for layer in model.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in model.hubert.encoder.layers[:11]:
            for param in layer.parameters():
                param.requires_grad = False

    elif freeze == "all":
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.hubert.parameters():
            param.requires_grad = False
    else:
        pass

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plot_name = (
        "upsample_augment_three_run_one_validate_three_layers"
        + str(freeze)
        + "_"
        + str(learning_rate)
        + "_"
    )
    checkpoint_path = os.path.join(
        "/home", "yyu", "model_checkpoints", plot_name + "_checkpoint.pt"
    )
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True, path=checkpoint_path)

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
        total_acc_val = 0
        total_loss_val = 0
        total_acc_train = 0
        total_loss_train = 0

        # Eval
        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                (
                    val_acc,
                    val_batch_loss,
                    val_output,
                    val_label,
                ) = train_audio_text_handler_model(
                    val_label, val_input, device, model, criterion
                )
                # Append results to the val lists
                val_output_list, val_label_list = append_to_list(
                    val_output.cpu(), val_label.cpu(), val_output_list, val_label_list
                )

                total_loss_val += val_batch_loss.item()
                total_acc_val += val_acc

        # Training
        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            (
                train_acc,
                train_batch_loss,
                train_output,
                train_label,
            ) = train_audio_text_handler_model(
                train_label, train_input, device, model, criterion
            )
            train_label = train_label.to(device)
            # Append results to the train lists
            train_output_list, train_label_list = append_to_list(
                train_output.cpu(),
                train_label.cpu(),
                train_output_list,
                train_label_list,
            )

            total_loss_train += train_batch_loss.item()
            total_acc_train += train_acc

            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(total_loss_val / len(val_data), model)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num + 1)
            break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_path))

        # Append to list
        train_loss_list.append(total_loss_train / len(train_data))
        train_acc_list.append(total_acc_train / len(train_data))
        val_loss_list.append(total_loss_val / len(val_data))
        val_acc_list.append(total_acc_val / len(val_data))

        # Generate plots
        # plot_name = "multi_upsample_augment_three_audio_freeze_all_val3_1-6_eleven_"
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

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .5f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .5f} \
                        | Val Loss: {total_loss_val / len(val_data): .5f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .5f}"
        )

        gc.collect()
        torch.cuda.empty_cache()


def train_audio_text_ablation(
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
    freeze,
    ablation_type,
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
    :param ablation_type: Type of ablation test.
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

    # Rules for freezing
    if freeze == "first_ten":
        for layer in model.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in model.hubert.encoder.layers[:11]:
            for param in layer.parameters():
                param.requires_grad = False

    # Rules for freezing
    if freeze == "first_ele":
        for layer in model.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in model.hubert.encoder.layers[:11]:
            for param in layer.parameters():
                param.requires_grad = False

    elif freeze == "all":
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.hubert.parameters():
            param.requires_grad = False
    else:
        pass

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plot_name = (
        "upsample_augment_three_layers"
        + str(freeze)
        + "_"
        + str(learning_rate)
        + "_"
        + ablation_type
        + "_"
    )
    checkpoint_path = os.path.join(
        "/home", "yyu", "model_checkpoints_ablation", plot_name + "_checkpoint.pt"
    )
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True, path=checkpoint_path)

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
        total_acc_val = 0
        total_loss_val = 0
        total_acc_train = 0
        total_loss_train = 0

        # Eval
        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                (
                    val_acc,
                    val_batch_loss,
                    val_output,
                    val_label,
                ) = train_audio_text_handler_model(
                    val_label, val_input, device, model, criterion
                )
                # Append results to the val lists
                val_output_list, val_label_list = append_to_list(
                    val_output.cpu(), val_label.cpu(), val_output_list, val_label_list
                )

                total_loss_val += val_batch_loss.item()
                total_acc_val += val_acc

        # Training
        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            (
                train_acc,
                train_batch_loss,
                train_output,
                train_label,
            ) = train_audio_text_handler_model(
                train_label, train_input, device, model, criterion
            )
            train_label = train_label.to(device)
            # Append results to the train lists
            train_output_list, train_label_list = append_to_list(
                train_output.cpu(),
                train_label.cpu(),
                train_output_list,
                train_label_list,
            )

            total_loss_train += train_batch_loss.item()
            total_acc_train += train_acc

            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(total_loss_val / len(val_data), model)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch_num + 1)
            break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_path))

        # Append to list
        train_loss_list.append(total_loss_train / len(train_data))
        train_acc_list.append(total_acc_train / len(train_data))
        val_loss_list.append(total_loss_val / len(val_data))
        val_acc_list.append(total_acc_val / len(val_data))

        # Generate plots
        # plot_name = "multi_upsample_augment_three_audio_freeze_all_val3_1-6_eleven_"
        gen_acc_plots(train_acc_list, val_acc_list, plot_name)
        gen_loss_plots(train_loss_list, val_loss_list, plot_name)
        gen_val_scatter_plot(val_output_list, val_label_list, plot_name)
        save_ablation_training_results(
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
        zip(
            train_loss_list,
            train_acc_list,
            val_loss_list,
            val_acc_list,
        )
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_loss_acc,
        columns=[
            "Train Loss",
            "Train Acc",
            "Val Loss",
            "Val Acc",
        ],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "plots",
            "para_tuning",
            "training_csv",
            plot_name + "loss_acc.csv",
        ),
        index=False,
    )

    list_of_tuples_output = list(
        zip(
            train_output_list,
            train_label_list,
            val_output_list,
            val_label_list,
        )
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_output,
        columns=[
            "Train Output",
            "Train Label",
            "Val Output",
            "Val Label",
        ],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "plots",
            "para_tuning",
            "training_csv",
            plot_name + "output_label.csv",
        ),
        index=False,
    )


# Save data to csv
def save_ablation_training_results(
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
        zip(
            train_loss_list,
            train_acc_list,
            val_loss_list,
            val_acc_list,
        )
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_loss_acc,
        columns=[
            "Train Loss",
            "Train Acc",
            "Val Loss",
            "Val Acc",
        ],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "plots",
            "ablation_correct",
            "training_csv",
            plot_name + "loss_acc.csv",
        ),
        index=False,
    )

    list_of_tuples_output = list(
        zip(
            train_output_list,
            train_label_list,
            val_output_list,
            val_label_list,
        )
    )
    loss_acc_df = pd.DataFrame(
        list_of_tuples_output,
        columns=[
            "Train Output",
            "Train Label",
            "Val Output",
            "Val Label",
        ],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "plots",
            "ablation_correct",
            "training_csv",
            plot_name + "output_label.csv",
        ),
        index=False,
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
    save_path = os.path.join(
        "/home", "yyu", "plots", "para_tuning", plot_name + "acc.png"
    )
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
    save_path = os.path.join(
        "/home", "yyu", "plots", "para_tuning", plot_name + "loss.png"
    )
    plt.savefig(save_path)
    plt.clf()


def gen_val_scatter_plot(val_output_list, val_label_list, plot_name):
    plt.figure()
    plt.scatter(val_label_list, val_output_list)
    plt.xlabel("Ground Truth Scores")
    plt.ylabel("Model Output Scores")
    plt.title("Model output and ground truth for validation")
    save_path = os.path.join(
        "/home", "yyu", "plots", "para_tuning", plot_name + "val_scatter.png"
    )
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
    model_name,
):
    """
    Evaluate accuracy for the model on vectorised audio/text data.
    :param model: Model to be used for deep learning.
    :param audio_feature_extractor: Pre-trained transformer to extract audio features.
    :param text_tokenizer: Tokenizer for text.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param test_absolute: Whether to use absolute test.
    :param model_name: Name of the model evaluated.
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
    output_list = []
    label_list = []
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

            # Append results to the train lists
            output_list, label_list = append_to_list(
                output.cpu(),
                test_label.cpu(),
                output_list,
                label_list,
            )

            # acc = (output.argmax(dim=1) == test_label).sum().item()

            acc = get_accuracy(output, test_label, test_absolute)
            total_acc_test += acc

    plot_name = str(model_name) + "_" + str("total") + "_"
    save_eval_results(
        output_list,
        label_list,
        plot_name,
    )
    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


def evaluate_audio_text_ablation(
    model,
    audio_feature_extractor,
    text_tokenizer,
    test_data,
    batch_size,
    vectorise,
    test_absolute,
    type,
    model_name,
):
    """
    Evaluate accuracy for the model on vectorised audio data.
    :param model: Model to be used for deep learning.
    :param audio_feature_extractor: Pre-trained transformer to extract audio features.
    :param text_tokenizer: Tokenizer for text.
    :param test_data: Dataframe to be tested.
    :param batch_size: Number of batches.
    :param test_absolute: Whether to use absolute test.
    :param type: Audio or text only.
    :param model_name: Name of the model evaluated
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
    output_list = []
    label_list = []
    with torch.no_grad():
        model.eval()
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            if type == "audio":
                # Audio
                input_values_tensor = test_input["audio"]["input_values"].squeeze(1)
                # Set audios to be zero
                input_values = torch.zeros(input_values_tensor.size()).to(device)
                # Text
                mask = test_input["text"]["attention_mask"].to(device)
                input_id = test_input["text"]["input_ids"].squeeze(1).to(device)
            elif type == "text":
                # Audio
                input_values = test_input["audio"]["input_values"].squeeze(1).to(device)
                # Text
                mask_tensor = test_input["text"]["attention_mask"].to(device)
                # Set mask and input id to be zero
                mask = torch.zeros(mask_tensor.size()).to(device)
                input_id_tensor = test_input["text"]["input_ids"].squeeze(1)
                input_id = torch.zeros(input_id_tensor.size(), dtype=torch.int).to(
                    device
                )

            output = model(input_values, input_id, mask)
            output = output.flatten()

            # Append results to the train lists
            output_list, label_list = append_to_list(
                output.cpu(),
                test_label.cpu(),
                output_list,
                label_list,
            )

            acc = get_accuracy(output, test_label, test_absolute)
            total_acc_test += acc

    plot_name = str(model_name) + "_" + str(type) + "_"
    save_eval_results(
        output_list,
        label_list,
        plot_name,
    )
    print(f"Test Accuracy: {total_acc_test / len(test_data): .5f}")


def save_eval_results(
    output_list,
    label_list,
    plot_name,
):
    """
    Save the results from model training.
    :param output_list: List of val output.
    :param label_list: List of tensors of val labels.
    :param plot_name: Name of the plot depending on model.
    :return: Save results to a csv.
    """
    list_of_tuples_output = list(zip(output_list, label_list))
    loss_acc_df = pd.DataFrame(
        list_of_tuples_output,
        columns=["Val Output", "Val Label"],
    )

    loss_acc_df.to_csv(
        os.path.join(
            "/home",
            "yyu",
            "plots",
            "ablation",
            plot_name + "output.csv",
        ),
        index=False,
    )


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
