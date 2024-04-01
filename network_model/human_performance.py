"""Test the accuracy of human performance on the confidence detection on the
test set. This is the upperbound of the model performance. We also test model
performance here.
"""

import pandas as pd
from model_utils import *


# home_dir is the location of folder
home_dir = os.path.join("/home", "youjing", "PersonalProjects", "confidence_css")
folder_path = os.path.join(home_dir, "data", "label_results")

# test dataset is the same as the human df.
crowdsourcing_results_test_df_path = os.path.join(
    folder_path,
    "Cleaned_Results_Test.csv",
)

human_df = pd.read_csv(
    os.path.join(
        folder_path,
        "Human_Labels.csv",
    )
)
# Specify path to outputs by the model
model_output_df = pd.read_csv(
    os.path.join(
        home_dir,
        "plots",
        "training_csv",
        ".csv",
    )
)


## Test human performance
# Get true scores and human scores
true_scores = (
    pd.read_csv(crowdsourcing_results_test_df_path)["average"].astype(float) - 2.5
)
human_scores = human_df["score4"].astype(float) - 2.5
count = get_accuracy(human_scores, true_scores, absolute=True)
print("Human accuracy rate is", count / human_scores.size)
# Get human icc
print(get_icc(human_scores, true_scores, icc_type="ICC(3,1)"))
# Get MSE
print(get_mse(human_scores, true_scores))

## Test model performance
train_output = model_output_df["Train Output"].astype(float)
train_labels = model_output_df["Train Label"].astype(float)
count = get_accuracy(train_output, train_labels, absolute=True)
print("Model Training accuracy rate is", count / train_labels.size)
# Get icc
print(get_icc(train_output, train_labels, icc_type="ICC(3,1)"))
# Get MSE
print(get_mse(train_output, train_labels))

val_output = model_output_df["Val Output"].astype(float)
val_labels = model_output_df["Val Label"].astype(float)
count = get_accuracy(val_output, val_labels, absolute=True)
print("Model Val accuracy rate is", count / val_labels.size)
# Get icc
print(get_icc(val_output, val_labels, icc_type="ICC(3,1)"))
# Get MSE
print(get_mse(val_output, val_labels))
