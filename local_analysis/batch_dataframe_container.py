"""Container for the analysis on results in batches."""

import numpy as np
import pandas as pd
import seaborn as sns
import krippendorff
from statsmodels.stats import inter_rater as irr
from scipy.stats import kendalltau, spearmanr, pearsonr
import math
from math import isnan
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


class BatchResultsDataframe:
    """
    A class for handling the csv containing the survey results.
    """

    def __init__(
        self,
        csv_path,
        samples_benchmark_csv_path,
        num_audios_per_HIT,
        num_workers_per_assignment,
        in_progress,
        hard_rule,
    ):
        """
        Initialise the dataframe class.
        :param csv_path: Path of csv of the survey results.
        :param samples_benchmark_csv_path: Path of the benchmarked audio results.
        :param num_audios_per_HIT: Number of audios present per HIT.
        :param num_workers_per_assignment: Number of workers working on an
                                    assignment.
        :param in_progress: Whether still in progress of rejecting workers.
        :param hard_rule: Whether to use hard rule to filter audios.
        """
        # Read the csv into a pandas dataframe
        self.csv_name = csv_path[:-4]
        # # If first time
        # self.batch_dataframe = pd.read_csv(csv_path)
        # If second / third time
        self.original_batch_dataframe = pd.read_csv(csv_path)
        # if still in the process of filtering and rejecting work
        if in_progress:
            self.batch_dataframe = self.original_batch_dataframe[
                self.original_batch_dataframe["AssignmentStatus"] == "Submitted"
            ]
        else:
            self.batch_dataframe = self.original_batch_dataframe[
                self.original_batch_dataframe["AssignmentStatus"] == "Approved"
            ]
        self.samples_benchmark_marked_dataframe = pd.read_csv(
            samples_benchmark_csv_path
        )
        self.num_audios_per_HIT = num_audios_per_HIT
        self.num_workers_per_assignment = num_workers_per_assignment
        self.answer_majority_rule_threshold = math.floor(
            self.num_workers_per_assignment / 2
        )

        # Set thresholds and error messages
        self.wrong_answer_threshold = 0.25
        self.rejection_message = "Wrong answers to more than 25% of test questions."
        self.hard_rule = hard_rule

        # Get dataframes after rejection and filtering
        self.dataframe_reject = self.reject_wrong_workers()
        self.dataframe_filter = self.filter_invalid_audios()
        self.dataframe_numbered = self.keep_number_const()
        self.dataframe_clean = self.clean_up_results()

    def get_num_assignments(self, dataframe):
        """
        Get the number of unique audios.
        :return: Integer number of unique assignments.
        """
        # Count total number of scores retained
        count = 0
        for i in range(1, self.num_audios_per_HIT + 1, 1):
            score_tag = "Answer.howMuch" + str(i)
            count += dataframe[score_tag].count()

        return count

    def get_audio_url_list(self, num_audios_per_HIT, ID_df):
        """
        :param num_audios_per_HIT: Number of audios in a HIT.
        :param ID_df: Dataframe grouped with the same Worker_ID.
        :return: A list of audio urls.
        """
        audio_url_list = []
        for i in range(1, num_audios_per_HIT + 1, 1):
            # Get audio urls
            audio_url_tag = "Input.audio_url_" + str(i)
            audio_url_list.append(ID_df[audio_url_tag].tolist())
        return audio_url_list

    def reject_wrong_workers(self):
        """Reject workers whose answer given to a benchmarked task is different
        from ground truth.
        """
        # Make a copy of the original dataframe
        batch_dataframe = self.batch_dataframe.copy()
        dict_of_Worker_ids = dict(iter(batch_dataframe.groupby("WorkerId")))

        num_assignments = self.get_num_assignments(batch_dataframe)

        for WorkerId, WorkerId_df in dict_of_Worker_ids.items():
            wrong_answer_count = 0
            test_question_count = 0
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = WorkerId_df[audio_url_tag].values[0]
                question_verify_tag = "Answer.questionVerify" + str(i) + ".on"
                speaker_number_verify_tag = (
                    "Answer.speakerNumberVerify" + str(i) + ".on"
                )
                # If a test sample is found
                if (
                    audio_url
                    in self.samples_benchmark_marked_dataframe["audio_url"].tolist()
                ):
                    # print("Found test audio.")
                    # Increment by 2 cos 2 questions
                    test_question_count += 2
                    true_question_verify = self.samples_benchmark_marked_dataframe.loc[
                        self.samples_benchmark_marked_dataframe.audio_url == audio_url,
                        "Answer.questionVerify",
                    ].values[0]
                    answer_question_verify = WorkerId_df.loc[
                        WorkerId_df[audio_url_tag] == audio_url, question_verify_tag
                    ].values[0]
                    true_speaker_number_verify = (
                        self.samples_benchmark_marked_dataframe.loc[
                            self.samples_benchmark_marked_dataframe.audio_url
                            == audio_url,
                            "Answer.speakerNumberVerify",
                        ].values[0]
                    )
                    answer_speaker_number_verify = WorkerId_df.loc[
                        WorkerId_df[audio_url_tag] == audio_url,
                        speaker_number_verify_tag,
                    ].values[0]
                    # If the answer is wrong
                    if answer_question_verify != true_question_verify:
                        wrong_answer_count += 1
                    if answer_speaker_number_verify != true_speaker_number_verify:
                        wrong_answer_count += 1
            # Calculate the rate of wrong answers
            if test_question_count > 0:
                rate_wrong_answers = wrong_answer_count / test_question_count
            else:
                rate_wrong_answers = 1
            # Reject if more than threshold of answers are wrong for the first few rounds
            if rate_wrong_answers > self.wrong_answer_threshold:
                batch_dataframe.loc[batch_dataframe.WorkerId == WorkerId, "Reject"] = (
                    self.rejection_message
                )
        # Approve the rest of the workers
        batch_dataframe.loc[batch_dataframe.Reject.isnull(), "Approve"] = "x"

        # Calculate the number of rejections
        number_of_rejections = batch_dataframe["Reject"].count()
        print(
            "Percentage of submissions rejected:",
            (number_of_rejections * self.num_audios_per_HIT) / (num_assignments),
        )
        saved_csv_name = self.csv_name + "_reject" + ".csv"
        batch_dataframe.to_csv(saved_csv_name, index=False)

        return batch_dataframe

    def get_index_positions(self, list_of_elems, element):
        """
        Find the positions of elements in a list.
        :param list_of_elems: list to be operated with.
        :param element: element to be found.
        :return: list of positions where the element has occured.
        """
        index_pos_list = []
        index_pos = 0
        while True:
            try:
                # Search for item in list from indexPos to the end of list
                index_pos = list_of_elems.index(element, index_pos)
                # Add the index position in list
                index_pos_list.append(index_pos)
                index_pos += 1
            except ValueError as e:
                break
        return index_pos_list

    def filter_invalid_audios(self):
        """Remove answers for audios which are deemed not question/multiple
        speakers and save to another file."""
        reject_csv_name = self.csv_name + "_reject" + ".csv"
        dataframe_reject = pd.read_csv(reject_csv_name)
        batch_dataframe_after_rejection = dataframe_reject[
            dataframe_reject["Approve"] == "x"
        ]

        # Create a dictionary of dataframes, where the key is the HITId and value
        # is the dataframe
        dict_of_HIT_ids = dict(iter(batch_dataframe_after_rejection.groupby("HITId")))
        num_assignments = self.get_num_assignments(batch_dataframe_after_rejection)

        wrong_question_type_count = 0
        wrong_speaker_number_count = 0
        not_question_audio_url_list = []
        multiple_speaker_audio_url_list = []

        for HITId, ID_df in dict_of_HIT_ids.items():
            # Iterate over all answers
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                # Get audio urls
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = ID_df[audio_url_tag].values[0]
                # Disagreement of whether a question
                question_verify_tag = "Answer.questionVerify" + str(i) + ".on"
                question_identity_list = ID_df[question_verify_tag].tolist()
                if self.hard_rule:
                    # Hard rule is majority law
                    if (
                        question_identity_list.count(False)
                        > self.answer_majority_rule_threshold
                    ):
                        wrong_question_type_count += 1
                        not_question_audio_url_list.append(audio_url)
                else:
                    # Soft rule where only discarded if all 3 agree
                    if (
                        len(set(question_identity_list)) == 1
                        and question_identity_list.count(False) > 0
                    ):
                        wrong_question_type_count += 1
                        not_question_audio_url_list.append(audio_url)

                # Disagreement of whether multiple speakers
                speaker_number_verify_tag = (
                    "Answer.speakerNumberVerify" + str(i) + ".on"
                )
                multiple_speakers_list = ID_df[speaker_number_verify_tag].tolist()
                if self.hard_rule:
                    # Hard rule is majority law
                    if (
                        multiple_speakers_list.count(True)
                        > self.answer_majority_rule_threshold
                    ):
                        wrong_speaker_number_count += 1
                        multiple_speaker_audio_url_list.append(audio_url)
                else:
                    # Soft rule where only discarded if all 3 agree
                    if (
                        len(set(multiple_speakers_list)) == 1
                        and multiple_speakers_list.count(True) > 0
                    ):
                        wrong_speaker_number_count += 1
                        multiple_speaker_audio_url_list.append(audio_url)

        final_df = batch_dataframe_after_rejection.copy()
        # Print urls of rejected audios
        print("rejected : not question", not_question_audio_url_list)
        print("rejected : multiple speakers", multiple_speaker_audio_url_list)

        # Replace urls to be discarded with np.nan
        for number_rejected_audio in multiple_speaker_audio_url_list:
            final_df = final_df.replace(number_rejected_audio, np.nan)
        for number_rejected_audio in not_question_audio_url_list:
            final_df = final_df.replace(number_rejected_audio, np.nan)

        # Replace invalid answers as nan
        for i in range(1, self.num_audios_per_HIT + 1, 1):
            # Get names of the tags
            audio_url_tag = "Input.audio_url_" + str(i)
            score_tag = "Answer.howMuch" + str(i)
            question_verify_tag = "Answer.questionVerify" + str(i) + ".on"
            speaker_number_verify_tag = "Answer.speakerNumberVerify" + str(i) + ".on"
            audio_url_list = final_df[audio_url_tag].tolist()
            # If the rejected audio is in this series
            if np.nan in audio_url_list:
                # print("Found in this column", i)
                # # Get the index, i.e. 0, 1, ... of the rejected audio
                # index_position_list = get_index_positions(audio_url_list,
                #                                           np.nan)
                mask = final_df[audio_url_tag].isnull()
                final_df[score_tag][mask] = np.nan
                final_df[question_verify_tag][mask] = np.nan
                final_df[speaker_number_verify_tag][mask] = np.nan

        saved_csv_name = self.csv_name + "_reject" + "_filtered" + ".csv"
        final_df.to_csv(saved_csv_name, index=False)

        # Count total number of scores retained
        count = 0
        for i in range(1, self.num_audios_per_HIT + 1, 1):
            score_tag = "Answer.howMuch" + str(i)
            count += final_df[score_tag].count()

        print(
            "Percentage of wrong question types:",
            str(wrong_question_type_count / num_assignments),
        )
        print(
            "Percentage of multiple speakers:",
            str(wrong_speaker_number_count / num_assignments),
        )
        print(
            "Percentage of data removed for being not questions and multiple speakers:",
            str(1 - count / num_assignments),
        )
        print(
            "Percentage of final data retained:",
            str(count / num_assignments),
        )
        print(
            "Percentage of final data retained compared to initially:",
            str(count / self.get_num_assignments(self.batch_dataframe)),
        )

        return final_df

    def keep_number_const(self):
        """
        Filter the csv so that for each audio there would be 3 scores. Save to another file.
        :return: A new csv with 3 answers for each audio
        """
        filter_csv_name = self.csv_name + "_reject_filtered" + ".csv"
        dataframe_reject = pd.read_csv(filter_csv_name)

        # Create a dictionary of dataframes, where the key is the HITId and value
        # is the dataframe
        dict_of_HIT_ids = dict(iter(dataframe_reject.groupby("HITId")))
        incomplete_HIT_id_list = []
        new_dict_of_HIT_ids = {}

        for HITId, ID_df in dict_of_HIT_ids.items():
            # Iterate over all answers
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                # Get audio urls
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = ID_df[audio_url_tag].values[0]
                # Only proceed if audio url not nan
                if audio_url != np.nan:
                    score_tag = "Answer.howMuch" + str(i)
                    scores = ID_df[score_tag].tolist()
                    # Discard if fewer than 3 scores
                    if len(scores) < 3:
                        incomplete_HIT_id_list.append(HITId)
                    else:
                        new_dict_of_HIT_ids[HITId] = ID_df.iloc[:3]
                    # # Only keep first 3 items
                    # elif len(scores) > 3:
                    #     ID_df = ID_df.iloc[:3]

        # # Using items() + dict comprehension to remove a dict.
        # new_dict = {key: val for key,
        #                          val in dict_of_HIT_ids.items() if key not in incomplete_HIT_id_list}

        final_df = pd.concat(ID_df for HITId, ID_df in new_dict_of_HIT_ids.items())

        saved_csv_name = self.csv_name + "_reject_filtered_numbered" + ".csv"
        final_df.to_csv(saved_csv_name, index=False)

        return final_df

    def clean_up_results(self):
        """
        Rearrange the csv so it only has the url and score. Save to another file.
        :return: A csv with columns containing the url, worker scores and average score.
        """
        filter_csv_name = self.csv_name + "_reject_filtered_numbered" + ".csv"
        dataframe_reject = pd.read_csv(filter_csv_name)

        # Create a dictionary of dataframes, where the key is the HITId and value
        # is the dataframe
        dict_of_HIT_ids = dict(iter(dataframe_reject.groupby("HITId")))
        new_dict = {}
        for HITId, ID_df in dict_of_HIT_ids.items():
            # Iterate over the urls
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = ID_df[audio_url_tag].values[0]
                # Only proceed if audio url not nan
                if audio_url != np.nan:
                    score_tag = "Answer.howMuch" + str(i)
                    # Only proceed if 3 or more answers
                    scores = ID_df[score_tag].tolist()
                    if len(scores) >= 3:
                        # Only take the first 3 answers if more than 3
                        cleaned_scores = scores[:3]
                        # The fourth value is the average score
                        cleaned_scores.append(ID_df[score_tag].mean())
                        new_dict[audio_url] = cleaned_scores

        final_df = pd.DataFrame.from_dict(new_dict, orient="index").reset_index()
        final_df = final_df.iloc[1:, :]
        # For total data
        final_df.columns = ["audio_url", "score1", "score2", "score3", "average"]
        final_df.dropna(subset=["audio_url"], inplace=True)

        saved_csv_name = (
            self.csv_name + "_reject_filtered_numbered" + "_cleaned" + ".csv"
        )
        final_df.to_csv(saved_csv_name, index=False)

        return final_df

    def get_batch_stats(self, dataframe):
        """
        Calculate the mean and standard deviation of all scores.
        :param dataframe: Dataframe on which to calculate mean and std.
        :return: Mean, std, maximum and minimum of all scores.
        """
        batch_dataframe_copy = dataframe.copy()
        dict_of_Worker_ids_normalised = dict(
            iter(batch_dataframe_copy.groupby("WorkerId"))
        )
        # Calculate the mean and std of all the scores
        total_scores_list = []
        for WorkerId, WorkerId_df in dict_of_Worker_ids_normalised.items():
            # Get all the scores the worker has given
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                score_tag = "Answer.howMuch" + str(i)
                score = WorkerId_df[score_tag]
                total_scores_list.append(score.values[0])
        total_scores_list_without_nan = [
            x for x in total_scores_list if isnan(x) == False
        ]
        batch_mean = np.mean(total_scores_list_without_nan)
        batch_std = np.std(total_scores_list_without_nan)
        batch_max = max(total_scores_list_without_nan)
        batch_min = min(total_scores_list_without_nan)
        return batch_mean, batch_std, batch_max, batch_min

    def get_stats_per_audio(self, dataframe):
        """
        Calculate the mean and std of the score per audio and plot histogram.
        :return: The mean and std of the score per audio. Plot the results.
        """
        # Now we calculate the mean and std per audio clip
        audio_url_std_dict = {}
        audio_url_mean_dict = {}

        dict_of_HIT_ids = dict(iter(dataframe.groupby("HITId")))

        for HITId, ID_df in dict_of_HIT_ids.items():
            # Iterate over all answers
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                score_tag = "Answer.howMuch" + str(i)
                scores = ID_df[score_tag].tolist()
                # Remove nan values
                scores = [x for x in scores if isnan(x) == False]

                # Mean of each task
                # Get audio urls
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = ID_df[audio_url_tag].values[0]
                mean = np.mean(scores)
                audio_url_mean_dict[audio_url] = mean

                # Standard deviation of each task
                std = np.std(scores)
                audio_url_std_dict[audio_url] = std

        # Plot the mean of confidence given per audio
        mean_list = list(audio_url_mean_dict.values())
        hist = sns.histplot(mean_list, color="cornflowerblue", kde=True, bins=25)
        plt.title("Histogram for Mean Scores", fontsize=20)
        plt.xlabel("Mean", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.savefig("plots/mean_audio.png")

        std_list = list(audio_url_std_dict.values())
        plt.figure()
        hist = sns.histplot(std_list, color="cornflowerblue", kde=True, bins=25)
        plt.title("Histogram for Standard Deviation of Scores", fontsize=20)
        plt.xlabel("Standard Deviation", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.savefig("plots/std_audio.png")

    def get_stats_per_worker(self, dataframe):
        """
        Calculate the mean and std of the score per audio and plot histogram.
        :return: The mean and std of the score per audio. Plot the results.
        """
        # Now we calculate the mean and std per audio worker
        worker_mean_dict = {}
        worker_std_dict = {}
        worker_number_dict = {}

        dict_of_Worker_ids = dict(iter(dataframe.groupby("WorkerId")))

        for WorkerId, WorkerId_df in dict_of_Worker_ids.items():
            worker_audio_count = 0
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                score_tag = "Answer.howMuch" + str(i)
                scores = WorkerId_df[score_tag].tolist()
                scores = [x for x in scores if isnan(x) == False]
                worker_audio_count += len(scores)
                # Mean of each task
                mean = np.mean(scores)
                worker_mean_dict[WorkerId] = mean

                # Standard deviation of each task
                std = np.std(scores)
                worker_std_dict[WorkerId] = std

            worker_number_dict[WorkerId] = worker_audio_count

        mean_list = list(worker_mean_dict.values())
        plt.hist(mean_list, bins=20)
        plt.xlabel("Mean Scores")
        plt.ylabel("Frequencies")
        plt.title("Histogram of Mean Scores Given for Each Worker")
        plt.savefig("plots/mean_worker.png")
        plt.show()

        std_list = list(worker_std_dict.values())
        plt.hist(std_list, bins=20)
        plt.xlabel("Standard Deviation of Scores")
        plt.ylabel("Frequencies")
        plt.title("Histogram of Standard Deviation of Scores Given for Each Worker")
        plt.savefig("plots/std_worker.png")
        plt.show()

        worker_number_list = list(worker_number_dict.values())
        plt.hist(worker_number_list, bins=20)
        plt.xlabel("Number of Audios Graded by Workers")
        plt.ylabel("Frequencies")
        plt.title("Histogram of Number of Audios Graded by Workers")
        plt.savefig("plots/number_audio_worker.png")
        plt.show()

    def get_agreement_percentage(
        self, worker_scores_categorised, worker_scores_next_categorised
    ):
        """
        Get the percentage agreement between two workers.
        :param worker_scores_categorised: List of categorised scores.
        :param worker_scores_next_categorised: List of categorised scores of next worker.
        :return: A floating number of percentage agreement.
        """
        # Calculate percentage agreement
        agreement_count = 0
        for i in range(len(worker_scores_categorised)):
            if worker_scores_categorised[i] == worker_scores_next_categorised[i]:
                agreement_count += 1
        return agreement_count / len(worker_scores_categorised)

    def get_icc(self, list1, list2, list3):
        """
        Get intraclass correlation score.
        :param list1: List of scores given by rater 1.
        :param list2: List of scores given by rater 2.
        :param list3: List of scores given by rater 3.
        :return: A floating number of ICC score.
        """
        # create DataFrame
        index_list = list(range(len(list1)))
        index_list_2 = index_list.copy()
        index_list_2.extend(index_list)
        index_list_2.extend(index_list)

        icc_df = pd.DataFrame(
            {
                "index": index_list_2,
                "rater": ["1"] * len(list1) + ["2"] * len(list1) + ["3"] * len(list1),
                "rating": [*list1, *list2, *list3],
            }
        )
        icc_results_df = pg.intraclass_corr(
            data=icc_df, targets="index", raters="rater", ratings="rating"
        )
        icc_value = icc_results_df.loc[icc_results_df.Type == "ICC3k", "ICC"]
        return icc_value

    def categorise_score(self, score):
        """
        Categorise the confidnece scores into 5 categories.
        :param score: Raw score input by user.
        :return: Categorised score.
        """
        if score == 5:
            cat_score = 4
        else:
            cat_score = math.floor(score)
        return cat_score

    def get_cohen_kappa(self, first_worker_cat, second_worker_cat, third_worker_cat):
        """
        Get the cohen kappa between workers.
        :param first_worker_cat: An array of scores of the first worker categorised.
        :param second_worker_cat: An array of scores of the second worker categorised.
        :param third_worker_cat: An array of scores of the third worker categorised.
        :return: A list of cohen kappas.
        """
        cohen_kappa_list = []
        # Get cohen kappa
        cohen_kappa_1 = cohen_kappa_score(first_worker_cat, second_worker_cat)
        cohen_kappa_list.append(cohen_kappa_1)
        cohen_kappa_2 = cohen_kappa_score(first_worker_cat, third_worker_cat)
        cohen_kappa_list.append(cohen_kappa_2)
        cohen_kappa_3 = cohen_kappa_score(second_worker_cat, third_worker_cat)
        cohen_kappa_list.append(cohen_kappa_3)
        return cohen_kappa_list

    def get_percentage_agreement(
        self, first_worker_cat, second_worker_cat, third_worker_cat
    ):
        """
        Get the percentage agreements between workers.
        :param first_worker_cat: An array of scores of the first worker categorised.
        :param second_worker_cat: An array of scores of the second worker categorised.
        :param third_worker_cat: An array of scores of the third worker categorised.
        :return: A list of percentage agreements.
        """
        percentage_agreement_list = []
        # Get percentage agreement
        percentage_agreement_1 = self.get_agreement_percentage(
            first_worker_cat, second_worker_cat
        )
        percentage_agreement_list.append(percentage_agreement_1)
        percentage_agreement_2 = self.get_agreement_percentage(
            first_worker_cat, third_worker_cat
        )
        percentage_agreement_list.append(percentage_agreement_2)
        percentage_agreement_3 = self.get_agreement_percentage(
            second_worker_cat, third_worker_cat
        )
        percentage_agreement_list.append(percentage_agreement_3)
        return percentage_agreement_list

    def get_kendall_coefficient(self, first_worker, second_worker, third_worker):
        """
        Calculate the kendall coefficient.
        :param first_worker: An array of scores of the first worker.
        :param second_worker: An array of scores of the second worker.
        :param third_worker: An array of scores of the third worker.
        :return: A list of kendall coefficients.
        """
        kendall_coefficient_list = []
        kendall_tau_1, _ = kendalltau(
            first_worker,
            second_worker,
        )
        kendall_coefficient_list.append(kendall_tau_1)

        kendall_tau_2, _ = kendalltau(
            first_worker,
            third_worker,
        )
        kendall_coefficient_list.append(kendall_tau_2)

        kendall_tau_3, _ = kendalltau(
            second_worker,
            third_worker,
        )
        kendall_coefficient_list.append(kendall_tau_3)
        return kendall_coefficient_list

    def get_spearman_coefficient(self, first_worker, second_worker, third_worker):
        """
        Calculate the spearman coefficient.
        :param first_worker: An array of scores of the first worker.
        :param second_worker: An array of scores of the second worker.
        :param third_worker: An array of scores of the third worker.
        :return: A list of spearman coefficients.
        """
        spearman_coefficient_list = []
        spearman_rho_1, _ = spearmanr(
            first_worker,
            second_worker,
        )
        spearman_coefficient_list.append(spearman_rho_1)

        spearman_rho_2, _ = spearmanr(
            first_worker,
            third_worker,
        )
        spearman_coefficient_list.append(spearman_rho_2)

        spearman_rho_3, _ = spearmanr(
            second_worker,
            third_worker,
        )
        return spearman_coefficient_list

    def get_pearson_coefficient(self, first_worker, second_worker, third_worker):
        """
        Calculate the pearson coefficient.
        :param first_worker: An array of scores of the first worker.
        :param second_worker: An array of scores of the second worker.
        :param third_worker: An array of scores of the third worker.
        :return: A list of pearson coefficients.
        """
        pearson_coefficient_list = []
        # Calculate Pearson's coefficient
        pearson_r_1, _ = pearsonr(
            first_worker,
            second_worker,
        )
        pearson_coefficient_list.append(pearson_r_1)

        pearson_r_2, _ = pearsonr(
            first_worker,
            third_worker,
        )
        pearson_coefficient_list.append(pearson_r_2)

        pearson_r_3, _ = pearsonr(
            second_worker,
            third_worker,
        )
        pearson_coefficient_list.append(pearson_r_3)
        return pearson_coefficient_list

    def get_irr_scores(self, dataframe):
        """
        Calculate the inter rater reliability scores index between different workers.
        :param dataframe: Dataframe on which to calculate irr scores.
        :return: A list of cohen kappa coefficients, krippendorff coefficients and icc between different workers.
        """
        dict_of_HIT_ids = dict(iter(dataframe.groupby("HITId")))

        # 2D array where column is the rater and row is the task.
        total_scores_list = []
        total_scores_categorised_list = []
        krippendorff_list = []
        icc_value_list = []
        fleiss_kappa_list = []
        for HIT, ID_df in dict_of_HIT_ids.items():
            # Iterate over the urls
            for i in range(1, self.num_audios_per_HIT + 1, 1):
                audio_url_tag = "Input.audio_url_" + str(i)
                audio_url = ID_df[audio_url_tag].values[0]
                # Only proceed if audio url not nan
                # if audio_url != np.nan:
                # if not isnan(audio_url):
                if audio_url == "" or pd.isnull(audio_url):
                    pass
                else:
                    score_tag = "Answer.howMuch" + str(i)
                    scores = ID_df[score_tag].tolist()
                    total_scores_list.append(scores)

                    # Categorise
                    scores_cat = [self.categorise_score(i) for i in scores]
                    total_scores_categorised_list.append(scores_cat)

        # Column is the rater (3 columns) and row is the task
        total_scores_matrix = np.array(total_scores_list).transpose()
        total_scores_categorised_matrix = np.array(
            total_scores_categorised_list
        ).transpose()

        # Convert to arrays
        first_worker, second_worker, third_worker = (
            total_scores_matrix[0],
            total_scores_matrix[1],
            total_scores_matrix[2],
        )

        print("first worker shape", first_worker.shape)

        first_worker_cat, second_worker_cat, third_worker_cat = (
            total_scores_categorised_matrix[0],
            total_scores_categorised_matrix[1],
            total_scores_categorised_matrix[2],
        )

        # Get cohen kappa
        cohen_kappa_list = self.get_cohen_kappa(
            first_worker_cat, second_worker_cat, third_worker_cat
        )

        # Get percentage agreement
        percentage_agreement_list = self.get_percentage_agreement(
            first_worker_cat, second_worker_cat, third_worker_cat
        )

        # Get krippendorff
        krippendorff_score = krippendorff.alpha(total_scores_categorised_matrix)
        krippendorff_list.append(krippendorff_score)

        ## Do not need categorisation
        # Get icc
        icc_value = self.get_icc(first_worker, second_worker, third_worker)
        icc_value_list.append(icc_value)

        # Get Fleiss' kappa
        dats, cats = irr.aggregate_raters(total_scores_matrix)
        fleiss_kappa = irr.fleiss_kappa(dats, method="fleiss")
        fleiss_kappa_list.append(fleiss_kappa)

        # Get Kendall's coefficient
        kendall_coefficient_list = self.get_kendall_coefficient(
            first_worker, second_worker, third_worker
        )

        # Calculate Spearman's coefficient
        spearman_coefficient_list = self.get_spearman_coefficient(
            first_worker, second_worker, third_worker
        )

        pearson_coefficient_list = self.get_pearson_coefficient(
            first_worker, second_worker, third_worker
        )

        return (
            cohen_kappa_list,
            percentage_agreement_list,
            krippendorff_list,
            icc_value_list,
            fleiss_kappa_list,
            kendall_coefficient_list,
            spearman_coefficient_list,
            pearson_coefficient_list,
        )
