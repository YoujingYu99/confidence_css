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
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft_train.csv",
)

print("start of application!")

# Read in individual csvs and load into a final dataframe
audio_text_df_train = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv, augment_audio=False
)

train_scores_list = audio_text_df_train["score"].tolist()


def count_scores_in_bins(train_scores_list):
    trains_scores_centered = [i - 2.5 for i in train_scores_list]
    first_bucket_count = 0
    second_bucket_count = 0
    third_bucket_count = 0
    fourth_bucket_count = 0
    fifth_bucket_count = 0
    for score in trains_scores_centered:
        if -2.5 <= score < -1.5:
            first_bucket_count += 1
        elif -1.5 <= score < -0.5:
            second_bucket_count += 1
        elif -0.5 <= score < 0.5:
            third_bucket_count += 1
        elif 0.5 <= score < 1.5:
            fourth_bucket_count += 1
        else:
            fifth_bucket_count += 1
    return (
        first_bucket_count,
        second_bucket_count,
        third_bucket_count,
        fourth_bucket_count,
        fifth_bucket_count,
    )


def plot_histogram_of_scores(scores_list, num_bins, plot_name):
    plt.figure()
    plt.hist(scores_list, bins=num_bins)
    plt.xlabel("Scores")
    plt.ylabel("Frequencies")
    plt.title("Histogram of " + plot_name)
    plt.savefig(os.path.join(home_dir, "plots", plot_name))
    plt.show()


# plot_histogram_of_scores(train_scores_list, num_bins=10,
#                          plot_name="training_dataset3")

print(count_scores_in_bins(train_scores_list))
