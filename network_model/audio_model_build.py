"""Use pre-trained HuBERT model on the audio for classification.
Extract the raw audio array and confidence score from the individual audio
classes. Then use this data to train the network for classification.
"""
from transformers import AutoFeatureExtractor
from models import *
from model_utils import *
from audio_features import *

folder_number = 6
# Decide on whether to load whole dataframe or individual ones.
loading_type = "many"
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
folder_path_dir = os.path.join(home_dir, "data_sheets", "features", str(folder_number))
total_audio_file = os.path.join(folder_path_dir, "test_model.csv")

print("start of application!")

# Read in individual csvs and load into a final dataframe
if loading_type == "many":
    audio_df = load_audio_and_score_from_folder(folder_path_dir)
    # Split to train, eval and test datasets.
    df_train, df_val, df_test = np.split(
        audio_df.sample(frac=1, random_state=42),
        [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
    )
elif loading_type == "single":
    audio_df = pd.read_csv(
        total_audio_file, converters={"audio_array": pd.eval}, encoding="ISO-8859-1"
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
batch_size = 1
audio_model = HubertClassifier()

# Train model

train_audio(
    audio_model, feature_extractor, df_train, df_val, LR, epochs, batch_size, vectorise
)
