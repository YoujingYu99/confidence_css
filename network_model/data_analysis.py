"""Analysis of the data for training, validation and test
"""
from transformers import AutoFeatureExtractor, BertTokenizer
from model_utils import *


# Decide whether to save the concatenated file to a single csv
save_to_single_csv = False
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
# Load feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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
audio_text_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
)

# Shuffling data again
audio_text_df = audio_text_df.sample(frac=1).reset_index(drop=True)
# Split to train, eval and test datasets.
df_train, df_val, df_test = np.split(
    audio_text_df.sample(frac=1, random_state=42),
    [int(0.8 * len(audio_text_df)), int(0.9 * len(audio_text_df))],
)

train_scores_list = df_train["score"].tolist()
val_scores_list = df_val["score"].tolist()
test_scores_list = df_test["score"].tolist()


def plot_histogram_of_scores(scores_list, num_bins, plot_name):
    plt.figure()
    plt.hist(scores_list, bins=num_bins)
    plt.xlabel("Scores")
    plt.ylabel("Frequencies")
    plt.title("Histogram of " + plot_name + " Scores")
    plt.savefig(os.path.join(home_dir, "plots", plot_name))
    plt.show()


plot_histogram_of_scores(train_scores_list, num_bins=10, plot_name="training_dataset3")
plot_histogram_of_scores(val_scores_list, num_bins=10, plot_name="validation_dataset3")
plot_histogram_of_scores(test_scores_list, num_bins=10, plot_name="test_dataset3")
