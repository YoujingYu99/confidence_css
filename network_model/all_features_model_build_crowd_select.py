"""Use model on the all features extracted from the audio for classification.
Extract the all features and confidence score to a csv. Then from the csv train the
model for confidence classification.
"""
from transformers import BertTokenizer
from model_utils import *
from models import *

# # Memory issues
# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)

# Decide whether to save the concatenated file to a single csv
save_to_single_csv = True
test_absolute = True

text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Load dataset
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft.csv",
)


print("start of application!")

# Choose features to focus on
features_to_use = [
    "interjecting_frequency",
    "energy",
    "energy_entropy",
    "spectral_centroids",
    "spectral_spread",
    "spectral_entropy",
    "spectral_rolloff",
    "spectral_contrast0",
    "spectral_contrast1",
    "spectral_contrast2",
    "spectral_contrast3",
    "spectral_contrast4",
    "spectral_contrast5",
    "spectral_contrast6",
    "zero_crossing_rate",
    "mfcc0",
    "mfcc1",
    "mfcc2",
    "mfcc3",
    "mfcc4",
    "mfcc5",
    "mfcc6",
    "mfcc7",
    "mfcc8",
    "mfcc9",
    "mfcc10",
    "mfcc11",
    "autocorrelation",
    "pitches",
    "tonnetz0",
    "tonnetz1",
    "tonnetz2",
    "tonnetz3",
    "tonnetz4",
    "tonnetz5",
    "pause_ratio",
    "repetition_rate",
]

# Read in individual csvs and load into a final dataframe
all_dict = load_select_features_and_score_from_crowdsourcing_results_selective(
    home_dir, crowdsourcing_results_df_path, features_to_use
)
# Get max number of rows
num_of_rows = get_num_rows(all_dict)
num_of_columns = len(features_to_use)

# Using items() + len() + list slicing
# Split dictionary by half
dict_train = dict(list(all_dict.items())[: math.floor(len(all_dict) * 0.8)])
dict_val = dict(
    list(all_dict.items())[
        math.floor(len(all_dict) * 0.8) : math.floor(len(all_dict) * 0.9)
    ]
)
dict_test = dict(list(all_dict.items())[math.floor(len(all_dict) * 0.9) :])
print(len(dict_train), len(dict_val), len(dict_test))


# Decide on Epoch and model
EPOCHS = 5
LR = 1e-6
batch_size = 3
num_workers = 4


# Train model
train_select_features(
    dict_train,
    dict_val,
    LR,
    EPOCHS,
    batch_size,
    num_workers,
    num_of_rows,
    num_of_columns,
    test_absolute,
)
evaluate_select_features(dict_test, batch_size, num_of_rows, num_of_columns)
