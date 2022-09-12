"""Use pre-trained HuBERT model on the audio for regression.
Extract the raw audio array and confidence score from the individual audio
classes. Then use this data to train the network for regression.
"""
from transformers import AutoFeatureExtractor
from model_utils import *


# Decide whether to save the concatenated file to a single csv
save_to_single_csv = False
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
# Path for crowdsourcing results
crowdsourcing_results_train_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft_train.csv",
)
crowdsourcing_results_val_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft_val.csv",
)
crowdsourcing_results_test_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft_test.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
audio_train_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_train_df_path, save_to_single_csv
)

audio_val_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_train_df_path, save_to_single_csv
)

audio_test_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_train_df_path, save_to_single_csv
)


# Training parameters
epochs = 500
LR = 5e-4
weight_decay = 5e-5
batch_size = 8
num_workers = 4

# Initialise audio model
# audio_model = HubertClassifier()
audio_model = CustomHUBERTModel()

# Train model
train_audio(
    audio_model,
    feature_extractor,
    audio_train_df,
    audio_val_df,
    LR,
    weight_decay,
    epochs,
    batch_size,
    vectorise,
    num_workers,
)

evaluate_audio(audio_model, audio_test_df, batch_size, feature_extractor, vectorise)
