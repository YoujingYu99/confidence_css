"""Inter-rater reliability analysis on using two or three scores from workers."""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from scipy.stats import kendalltau, spearmanr, pearsonr
import pingouin as pg
from statsmodels.stats import inter_rater as irr

# Label and title size
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20


class TwoScoreResultsDataframe:
    """
    A class for calculating the inter-rater reliability scores between two workers.
    """

    def __init__(
        self,
        csv_path,
    ):
        three_workers_df = pd.read_csv(csv_path)
        self.three_workers_df = three_workers_df
        self.two_workers_df = self.three_to_two_workers(three_workers_df)
        self.three_average_compare_df = self.three_average_df(three_workers_df)
        self.three_two_compare_df = self.three_two_df(three_workers_df)
        self.test_df = self.get_test_df(three_workers_df)
        self.three_average_human_compare_df = self.three_average_human_df(self.test_df)
        self.three_two_human_compare_df = self.three_two_human_df(self.test_df)

    def get_test_df(self, three_workers_df):
        test_human_df = pd.read_csv(
            "csv/Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test_yyu.csv"
        )

        audio_url_list = test_human_df["audio_url"]
        test_three_df = three_workers_df[
            three_workers_df["audio_url"].isin(audio_url_list)
        ]

        result = pd.merge(test_three_df, test_human_df, on="audio_url")

        return result

    def take_two_from_row(self, row):
        """
        Select the two most similar scores from the row and take average, then shift by 2.5.
        :param row: Pandas dataframe row
        :return: An average score.
        """
        # Take two scores that agree better with each other
        diff_one_two = abs(row["score1"] - row["score2"])
        diff_two_three = abs(row["score2"] - row["score3"])
        diff_one_three = abs(row["score1"] - row["score3"])
        diff_list = [diff_one_two, diff_two_three, diff_one_three]
        val, idx = min((val, idx) for (idx, val) in enumerate(diff_list))
        if idx == 0:
            score1, score2 = row["score1"], row["score2"]

        elif idx == 1:
            score1, score2 = row["score2"], row["score3"]
        else:
            score1, score2 = row["score1"], row["score3"]

        return score1, score2

    def three_to_two_workers(self, three_workers_df):
        audio_url_list = []
        score1_list = []
        score2_list = []
        for index, row in three_workers_df.iterrows():
            score1, score2 = self.take_two_from_row(row)
            score1_list.append(score1)
            score2_list.append(score2)
            audio_url_list.append(row["audio_url"])

        two_workers_df = pd.DataFrame(
            np.column_stack([audio_url_list, score1_list, score2_list]),
            columns=["audio_url", "score1", "score2"],
        )

        return two_workers_df

    def three_average_df(self, three_workers_df):
        """
        Get the dataframe where score1 is the original workers and score 2 average over three workers.
        :param three_workers_df: Original df with three scores
        :return:
        """
        score1_list = []
        score2_list = []
        for index, row in three_workers_df.iterrows():
            score1_list.append(row["score1"])
            score1_list.append(row["score2"])
            score1_list.append(row["score3"])
            for i in range(3):
                score2_list.append(row["average"])

        three_average_df = pd.DataFrame(
            np.column_stack([score1_list, score2_list]),
            columns=["score1", "score2"],
        )

        return three_average_df

    def three_average_human_df(self, test_df):
        """
        Get the dataframe where score1 is the original workers and score 2 average over three workers.
        :param test_df: Original df with three scores and human score.
        :return:
        """
        score1_list = []
        score2_list = []
        for index, row in test_df.iterrows():
            score1_list.append(row["score4"])
            score2_list.append((row["score1"] + row["score2"] + row["score3"]) / 3)

        three_average_human_df = pd.DataFrame(
            np.column_stack([score1_list, score2_list]),
            columns=["score1", "score2"],
        )

        return three_average_human_df

    def three_two_df(self, three_workers_df):
        """
        Get the dataframe where score1 is the original workers and score 2 average over two workers.
        :param three_workers_df: Original df with three scores.
        :return:
        """
        score1_list = []
        score2_list = []
        for index, row in three_workers_df.iterrows():
            score1_list.append(row["score1"])
            score1_list.append(row["score2"])
            score1_list.append(row["score3"])
            close1, close2 = self.take_two_from_row(row)
            for i in range(3):
                score2_list.append((close1 + close2) / 2)

        three_two_df = pd.DataFrame(
            np.column_stack([score1_list, score2_list]),
            columns=["score1", "score2"],
        )

        return three_two_df

    def three_two_human_df(self, test_df):
        """
        Get the dataframe where score1 is the human score and score 2 average over two workers.
        :param test_df: Original df with three scores and human score.
        :return:
        """
        score1_list = []
        score2_list = []
        for index, row in test_df.iterrows():
            score1_list.append(row["score4"])
            close1, close2 = self.take_two_from_row(row)
            score2_list.append((close1 + close2) / 2)

        three_two_human_df = pd.DataFrame(
            np.column_stack([score1_list, score2_list]),
            columns=["score1", "score2"],
        )

        return three_two_human_df

    def get_icc(self, list1, list2):
        """
        Get intraclass correlation score.
        :param list1: List of scores given by rater 1.
        :param list2: List of scores given by rater 2.
        :return: A floating number of ICC score.
        """
        # create DataFrame
        index_list = list(range(len(list1)))
        index_list_2 = index_list.copy()
        index_list_2.extend(index_list)

        icc_df = pd.DataFrame(
            {
                "index": index_list_2,
                "rater": ["1"] * len(list1) + ["2"] * len(list1),
                "rating": [*list1, *list2],
            }
        )
        icc_results_df = pg.intraclass_corr(
            data=icc_df, targets="index", raters="rater", ratings="rating"
        )
        icc_value_df = icc_results_df.loc[icc_results_df.Type == "ICC3k", "ICC"]
        icc_value = icc_value_df.to_numpy()[0]
        return icc_value

    def test_accuracy(self, output, actual, absolute):
        """
        Testify whether the output is accurate.
        :param output: Score tensor output by model.
        :param actual: Actual score tensor.
        :param absolute: Whether to test with absolute value
        :return: Number of accurate predicitons
        """
        output_list = output.tolist()
        actual_list = actual.tolist()
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

    def test_human_accuracy(self, test_df):
        human_series = test_df["score1"]
        ave_series = test_df["score2"]

        acc = (
            self.test_accuracy(human_series, ave_series, absolute=True)
            / human_series.size
        )

        return acc

    def get_irr_scores(self, dataframe):
        """
        Calculate the inter rater reliability scores index between different workers.
        :param dataframe: Dataframe on which to calculate irr scores.
        :return: A list of icc, kendall's and Fless kappa.
        """

        # Row is the rater (2 rows) and column is the task
        first_worker_scores_list = dataframe["score1"].tolist()
        first_worker_scores_list = [float(i) for i in first_worker_scores_list]
        second_worker_scores_list = dataframe["score2"].tolist()
        second_worker_scores_list = [float(i) for i in second_worker_scores_list]
        total_scores_matrix = np.column_stack(
            (first_worker_scores_list, second_worker_scores_list)
        )

        ## Do not need categorisation
        # Get icc
        icc_value = self.get_icc(first_worker_scores_list, second_worker_scores_list)
        # Get Fleiss' kappa
        dats, cats = irr.aggregate_raters(total_scores_matrix)
        fleiss_kappa = irr.fleiss_kappa(dats, method="fleiss")
        # Get Kendall Taud
        kendall_tau, _ = kendalltau(first_worker_scores_list, second_worker_scores_list)
        # Spearman's rho
        spearman_rho, _ = spearmanr(first_worker_scores_list, second_worker_scores_list)
        # Pearson's r
        pearson_r, _ = pearsonr(first_worker_scores_list, second_worker_scores_list)

        return (icc_value, fleiss_kappa, kendall_tau, spearman_rho, pearson_r)

    def get_distribution(self, dataframe):
        """
        Plot histogram of 5 bins.
        :param dataframe: Dataframe of scores.
        :return: Saved histogram.
        """
        input_list = dataframe["average"].tolist()
        one_count = 0
        two_count = 0
        three_count = 0
        four_count = 0
        five_count = 0

        for score in input_list:
            if score < 1:
                one_count += 1
            elif 1 <= score < 2:
                two_count += 1
            elif 2 <= score < 3:
                three_count += 1
            elif 3 <= score < 4:
                four_count += 1
            elif 4 <= score <= 5:
                five_count += 1

        return one_count, two_count, three_count, four_count, five_count

    def plot_distribution(self, dataframe):
        hist = sns.histplot(
            data=dataframe, x="average", color="cornflowerblue", kde=True, bins=30
        )
        plt.title("Histogram of Score Distribution", fontsize=20)
        plt.xlabel("Score", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        save_path = os.path.join(
            "/home", "yyu", "plots", "dataset_analysis", "scores_hist.png"
        )
        plt.savefig(save_path)


home_dir = os.path.join("/home", "yyu")
csv_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned.csv",
)

original_df = TwoScoreResultsDataframe(csv_path)

icc_value, fleiss_kappa, kendall_tau, spearman_rho, pearson_r = (
    original_df.get_irr_scores(original_df.three_two_human_compare_df)
)
print(icc_value, fleiss_kappa, kendall_tau, spearman_rho, pearson_r)

print(original_df.get_distribution(original_df.three_workers_df))
