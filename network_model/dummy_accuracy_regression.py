"""Dummy accuracy test for minimum required accuracy."""
import pandas as pd

from model_utils import load_audio_and_score_from_crowdsourcing_results, get_icc
import os
import numpy as np
import random

save_to_single_csv = False

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned.csv",
)

model_training_results_df_path = os.path.join(
    home_dir,
    "plots",
    "training_csv",
    "audio_text_upsample_two_augment_training_result.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
audio_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_df_path,
    save_to_single_csv,
    augment_audio=False,
    two_scores=True,
)

true_scores = audio_df["score"].tolist()
# Generate random scores
random_list = []
for i in range(len(true_scores)):
    random_list.append(random.uniform(-2.5, 2.5))


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
            if actual_list[i] - 0.5 <= output_list[i] <= actual_list[i] + 0.5:
                count += 1
        else:
            if actual_list[i] * 0.8 <= output_list[i] <= actual_list[i] * 1.2:
                count += 1
    return count


## Test accuracy
# accuracy = test_accuracy(random_list, true_scores, absolute=True) / len(random_list)
# print("Random classifier accuracy is", accuracy)

## Test ICC
# print(get_icc(random_list, true_scores, icc_type="ICC(3,1)"))


def calculate_mse(output_list, actual_list):
    """
    Calculate MSE between two lists.
    :param output_list: Score list output by model.
    :param actual_list: Actual score list.
    :return: MSE value.
    """
    mse = np.mean((np.array(actual_list) - np.array(output_list)) ** 2)
    return mse


## Test MSE
print(calculate_mse(random_list, true_scores))


## Test model ICC
mode_result = pd.read_csv(
    model_training_results_df_path,
    usecols=["Train Output", "Train Label", "Val Output", "Val Label"],
)

train_icc = get_icc(
    mode_result["Train Output"].tolist(),
    mode_result["Train Label"].tolist(),
    icc_type="ICC(3,1)",
)
val_icc = get_icc(
    mode_result["Val Output"].tolist(),
    mode_result["Val Label"].tolist(),
    icc_type="ICC(3,1)",
)
