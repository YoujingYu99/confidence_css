"""Dummy accuracy test for minimum required accuracy."""
from model_utils import load_audio_and_score_from_crowdsourcing_results
import os
import random

save_to_single_csv = False

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

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


true_scores = audio_df["score"].tolist()
# Generate random scores
random_list = []
for i in range(len(true_scores)):
    random_list.append(random.uniform(-2.5, 2.5))

print(random_list)


def test_accuracy(output_list, actual_list, absolute):
    """
    Testify whether the output is accurate.
    :param output_list: Score list output by model.
    :param actual_list: Actual score list.
    :param absolute: Whether to test with absolute value
    :return: Number of accurate predicitons
    """
    count = 0
    for i in range(len(output_list)):
        # If test by absolute value
        if absolute:
            if actual_list[i] - 0.2 <= output_list[i] <= actual_list[i] + 0.2:
                count += 1
        else:
            if actual_list[i] * 0.8 <= output_list[i] <= actual_list[i] * 1.2:
                count += 1
    return count


accuracy = test_accuracy(random_list, true_scores, absolute=True) / len(random_list)

print("Random classifier accuracy is", accuracy)
