"""Use pre-trained BERT model on the text for regression.
Extract the text and confidence score to a csv. Then from the csv train the
BERT model for confidence regression.
"""
from transformers import BertTokenizer
from model_utils import *
from models import *

# # Memory issues
# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)

# Decide whether to save the concatenated file to a single csv
save_to_single_csv = False
test_absolute = True

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

# Read in individual csvs and load into a final dataframe
text_train_df = load_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_train_df_path,
    save_to_single_csv,
    augment_text=True,
    two_scores=True,
)

text_val_df = load_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_val_df_path,
    save_to_single_csv,
    augment_text=False,
    two_scores=True,
)

text_test_df = load_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_test_df_path,
    save_to_single_csv,
    augment_text=False,
    two_scores=True,
)

# Decide on Epoch and model
epochs = 500
LR = 1e-6
weight_decay = 5e-5
batch_size = 16
num_workers = 4
accum_iter = 4

# Initialise model
# text_model = BertClassifier()
text_model = CustomBERTSimpleModel()

# Train model
train_text(
    text_model,
    text_tokenizer,
    text_train_df,
    text_val_df,
    LR,
    weight_decay,
    epochs,
    batch_size,
    num_workers,
    test_absolute,
    accum_iter,
)


evaluate_text(text_model, text_test_df, text_tokenizer, batch_size, test_absolute)
