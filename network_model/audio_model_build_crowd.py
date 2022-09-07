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
featuers_folder_path_dir = os.path.join(home_dir, "data_sheets", "features")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
audio_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
)
# Split to train, eval and test datasets.
df_train, df_val, df_test = np.split(
    audio_df.sample(frac=1, random_state=42),
    [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
)


print(len(df_train), len(df_val), len(df_test))
print(audio_df.head())


# Training parameters
epochs = 500
LR = 5e-5
batch_size = 8
num_workers = 4

# Initialise audio model
# audio_model = HubertClassifier()
audio_model = CustomHUBERTSimpleModel()

# Train model

train_audio(
    audio_model,
    feature_extractor,
    df_train,
    df_val,
    LR,
    epochs,
    batch_size,
    vectorise,
    num_workers,
)

evaluate_audio(audio_model, df_test, batch_size, feature_extractor, vectorise)
