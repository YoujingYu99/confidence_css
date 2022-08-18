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

text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Load dataset
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
all_dict = load_all_features_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path
)

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

# num_rows = get_num_rows(all_dict)
# print("max number of rows", num_rows)

# Decide on Epoch and model
EPOCHS = 5
LR = 1e-6
batch_size = 3
num_workers = 4


# Train model
train_all_features(dict_train, dict_val, LR, EPOCHS, batch_size, num_workers)
evaluate_all_features(dict_test, batch_size)
