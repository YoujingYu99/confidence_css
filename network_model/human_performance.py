"""Test the accuracy of human performance on the confidence detection on the
test set. This is the upperbound of the model performance
"""
from model_utils import *


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
crowdsourcing_results_test_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test.csv",
)

audio_text_test_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_test_df_path,
    save_to_single_csv=False,
    augment_audio=False,
    two_scores=True,
)

human_df = pd.read_csv(
    os.path.join(
        home_dir,
        "data_sheets",
        "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test_yyu.csv",
    )
)

# Get two series of scores
true_scores = audio_text_test_df["score"].astype(float)
human_scores = human_df["score4"].astype(float) - 2.5

count = test_accuracy(human_scores, true_scores, absolute=True)
print("Human accuracy rate is", count / human_scores.size)
