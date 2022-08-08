"""Use pre-trained HuBERT model on the audio for classification.
Extract the raw audio array and confidence score from the individual audio
classes. Then use this data to train the network for classification.
"""
from transformers import AutoFeatureExtractor
from models import *
from model_utils import *
from audio_features import *

folder_number = 5
# Decide whether to save the concatenated file to a single csv
save_to_single_csv = True
# Decide on whether to load whole dataframe or individual ones.
loading_type = "many"
# Decide whether to use audio only files or all feature files
file_type = "audio_only"
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
featuers_folder_path_dir = os.path.join(
    home_dir, "data_sheets", "features", str(folder_number)
)
# audio_only_folder_path_dir = os.path.join(
#     home_dir, "data_sheets", "features_audio_array", str(folder_number)
# )
audio_only_folder_path_dir = os.path.join(
    home_dir, "data_sheets", "features_audio_array", "5_test"
)
## TODO: these files do not exist yet due to failure in loading large csvs
all_features_total_audio_file_path = os.path.join(
    featuers_folder_path_dir, "all_features_all_model.csv"
)
## TODO: these files do not exist yet due to failure in loading large csvs
audio_only_total_audio_file_path = os.path.join(
    featuers_folder_path_dir, "audio_only_all_model.csv"
)

print("start of application!")

# Read in individual csvs and load into a final dataframe
if loading_type == "many":
    if file_type == "audio_only":
        audio_df = load_audio_and_score_from_folder(
            audio_only_folder_path_dir, file_type, save_to_single_csv
        )
        # Split to train, eval and test datasets.
        df_train, df_val, df_test = np.split(
            audio_df.sample(frac=1, random_state=42),
            [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
        )
    elif file_type == "all_features":
        audio_df = load_audio_and_score_from_folder(
            featuers_folder_path_dir, file_type, save_to_single_csv
        )
        # Split to train, eval and test datasets.
        df_train, df_val, df_test = np.split(
            audio_df.sample(frac=1, random_state=42),
            [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
        )

# Read in a single large scv file containing all the information of a whole batch
elif loading_type == "single":
    if file_type == "audio_only":
        audio_df = pd.read_csv(
            all_features_total_audio_file_path,
            converters={"audio_array": pd.eval},
            encoding="ISO-8859-1",
        )
    elif file_type == "all_features":
        audio_df = pd.read_csv(
            audio_only_total_audio_file_path,
            converters={"audio_array": pd.eval},
            encoding="ISO-8859-1",
        )
    # Split to train, eval and test datasets.
    df_train, df_val, df_test = audio_df.random_split([0.8, 0.1, 0.1], random_state=123)

print(len(df_train), len(df_val), len(df_test))
print(audio_df.head())

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Training parameters
epochs = 5
LR = 1e-6

batch_size = 3
audio_model = HubertClassifier()

# Train model

train_audio(
    audio_model, feature_extractor, df_train, df_val, LR, epochs, batch_size, vectorise
)
