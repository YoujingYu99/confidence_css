"""Use pre-trained HuBERT model on the audio for classification.
Extract the raw audio array and confidence score from the individual audio
classes. Then use this data to train the network for classification.
"""
import os
import pandas as pd
import torch
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from model_utils import *
from models import *
from feature_extraction.features import *

# Memory issues
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


# loading model and tokenizer
model_name_or_path = "facebook/hubert-base-ls960"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path, num_labels=10, finetuning_task="wav2vec2_clf",
)
setattr(config, "pooling_mode", pooling_mode)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)


def load_audio_and_score_from_folder(home_dir, folder_path_list):
    """
    Load the confidence score and audio array from the csv files.
    :param home_dir: Primary directory.
    :param folder_path_list: Path of the folder of csvs
    :return:
    """
    file_dir = os.path.join(home_dir, folder_path_list)
    audio_list = []
    score_list = []
    max_length = 0
    for filename in os.listdir(file_dir):
        total_df = pd.read_csv(filename)
        audio_list.append(total_df["audio_array"])
        score_list.append(total_df["score"])
        # Update max length if a longer audio occurs
        if len(total_df["audio_array"]) > max_length:
            max_length = len(total_df["audio_array"])

    result = feature_extractor(audio_list, sampling_rate=16000)
    result["labels"] = score_list


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Create folder path list
folder_path_list = []
for i in range(8):
    folder_path = os.path.join(home_dir, "extracted_audio", str(i))
    folder_path_list.append(folder_path)

