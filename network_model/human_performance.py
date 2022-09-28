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

# count = test_accuracy(human_scores, true_scores, absolute=True)
# print("Human accuracy rate is", count / human_scores.size)


def test_accuracy_score(output, actual, absolute, margin):
    """
    Testify whether the output is accurate.
    :param output: Score tensor output by model.
    :param actual: Actual score tensor.
    :param absolute: Whether to test with absolute value.
    :param margin: Margin for threshold.
    :return: Number of accurate predicitons
    """
    output_list = output.tolist()
    actual_list = actual.tolist()
    count = 0
    for i in range(len(output_list)):
        # If test by absolute value
        if absolute:
            if actual_list[i] - margin <= output_list[i] <= actual_list[i] + margin:
                count += 1
    return count


def plot_accuracy_boundary(human_scores, true_scores):
    """
    Test the accuracy with the margin sensitivity.
    :param human_scores: Scores given by human.
    :param true_scores: Actual scores.
    :return: Plot saved to local path.
    """
    margin_array = np.arange(0, 2.5, 0.001)
    accuracy_list = []
    for margin in margin_array:
        accuracy = (
            test_accuracy_score(human_scores, true_scores, absolute=True, margin=margin)
            / human_scores.size
        )
        accuracy_list.append(accuracy)

    plt.figure()
    plt.plot(margin_array, accuracy_list)
    plt.xlabel("Margin")
    plt.ylabel("Accuracy")
    plt.title("Plot of Accuracy against Margin")
    save_path = os.path.join("/home", "yyu", "plots", "acc_margin.png")
    plt.savefig(save_path)


## Plot the accuracy with respect to the margins
# plot_accuracy_boundary(human_scores, true_scores)

# Get human icc
# print(get_icc(human_scores, true_scores, icc_type="ICC(3,1)"))
# Get MSE
# print(calculate_mse(human_scores, true_scores))

print(calculate_mse(human_scores, true_scores))
