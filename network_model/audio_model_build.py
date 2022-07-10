"""Use pre-trained HuBERT model on the audio for classification.
Extract the raw audio array and confidence score from the individual audio
classes. Then use this data to train the network for classification.
"""
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor
from transformers import (
    AutoConfig,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
)
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel,
)

import torch
from models import *
from model_utils import *
from audio_features import *

# Memory issues
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))


# # loading model and tokenizer
# model_name_or_path = "facebook/hubert-base-ls960"
# pooling_mode = "mean"
#
# # config
# config = AutoConfig.from_pretrained(
#     model_name_or_path, num_labels=10, finetuning_task="wav2vec2_clf",
# )
# setattr(config, "pooling_mode", pooling_mode)
#
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


free_gpu_cache()

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
folder_path_dir = os.path.join(home_dir, "data_sheets", "features", "6")


audio_df = load_audio_and_score_from_folder(folder_path_dir)
# print(audio_df.head())
df_train, df_val, df_test = np.split(
    audio_df.sample(frac=1, random_state=42),
    [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
)
print(len(df_train), len(df_val), len(df_test))

# model_checkpoint = "facebook/wav2vec2-base"
# num_labels = 10
# batch_size = 10
# audio_model = AutoModelForAudioClassification.from_pretrained(
#     model_checkpoint,
#     num_labels=num_labels
# )
#
# model_name = model_checkpoint.split("/")[-1]
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# audio_tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# Decide on Epoch and model
EPOCHS = 5
LR = 1e-6
audio_model = HubertClassifier()
# Train model
train_audio(audio_model, feature_extractor, df_train, df_val, LR, EPOCHS)
# # home_dir is the location of script
# home_dir = os.path.join("/home", "yyu")
#
# # Create folder path list
# folder_path_list = []
# for i in range(8):
#     folder_path = os.path.join(home_dir, "extracted_audio", str(i))
#     folder_path_list.append(folder_path)
