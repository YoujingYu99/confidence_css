"""Use pre-trained Bert and HuBERT models on the audio and text for regression.
Extract the raw audio array, transcription and confidence score from the individual audio
classes. Then use this data to train the network for regression.
"""
import os.path

import pandas as pd
from transformers import AutoFeatureExtractor, BertTokenizer
from model_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Decide whether to save the concatenated file to a single csv
save_to_single_csv = False
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
two_scores = True
test_absolute = True
# Load feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_train_df_path = os.path.join(
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)
crowdsourcing_results_val_df_path = os.path.join(
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)
crowdsourcing_results_test_df_path = os.path.join(
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)


print("start of application!")
generate_train_data_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_train_df_path,
    augment_audio=True,
    two_scores=two_scores,
)


# Read in individual csvs and load into a final dataframe
audio_text_train_df = pd.read_csv(
    os.path.join(home_dir, "data_sheets", "audio_text_upsampled_augmented.csv")
)


audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_val_df_path,
    save_to_single_csv,
    augment_audio=False,
    two_scores=two_scores,
)

audio_text_test_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_test_df_path,
    save_to_single_csv,
    augment_audio=False,
    two_scores=two_scores,
)


# Training parameters
epochs = 300
LR = 5e-5
weight_decay = 1e-6
batch_size = 2
num_workers = 4
accum_iter = 4

# Initialise audio model
# audio_model = HubertClassifier()
multimodel = CustomMultiModelSimplePooled()

# Train model
train_audio_text(
    multimodel,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_train_df,
    audio_text_val_df,
    LR,
    weight_decay,
    epochs,
    batch_size,
    num_workers,
    accum_iter,
    vectorise,
    test_absolute,
)

evaluate_audio_text(
    multimodel,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_test_df,
    batch_size,
    vectorise,
    test_absolute,
)
