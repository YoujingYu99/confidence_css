"""Analyse the batch results."""

import os
from batch_dataframe_container import *

home_dir = os.path.join("/home", "yyu")
# csv_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
#     "Batch_4799159_batch_results_complete.csv",
# )
# samples_benchmark_csv_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
#     "Samples_Benchmark_200_Marked.csv",
#     "Benchmark_Samples.csv",
# )
csv_path = os.path.join(
    home_dir,
    "Label_Results",
    "Complete_Results.csv",
)
samples_benchmark_csv_path = os.path.join(
    home_dir,
    "Label_Results",
    "Benchmark_Samples.csv",
)


num_audios_per_HIT = 12
num_workers_per_assignment = 3


original_results = BatchResultsDataframe(
    csv_path,
    samples_benchmark_csv_path,
    num_audios_per_HIT,
    num_workers_per_assignment,
    in_progress=False,
    hard_rule=False,
)

# Before normalisation
batch_mean, batch_std, batch_max, batch_min = original_results.get_batch_stats(
    dataframe=original_results.dataframe_numbered
)
(
    cohen_kappa_list,
    percentage_agreement_list,
    krippendorff_list,
    icc_list,
    fleiss_kappa_list,
    kendall_coefficient_list,
    spearman_coefficient_list,
    pearson_coefficient_list,
) = original_results.get_irr_scores(dataframe=original_results.dataframe_numbered)


print("average ICC", np.mean(icc_list))
print("average cohen kappa", np.mean(cohen_kappa_list))
print("average percentage agreement", np.mean(percentage_agreement_list))
print("average krippendorff", np.mean(krippendorff_list))
print("average fleiss kappa", np.mean(fleiss_kappa_list))
print("average kendall's coefficient", np.mean(kendall_coefficient_list))
print("average spearman coefficient", np.mean(spearman_coefficient_list))
print("average pearson coefficient", np.mean(pearson_coefficient_list))
print("batch mean", batch_mean)
print("batch std", batch_std)
print("batch max", batch_max)
print("batch min", batch_min)


original_results.get_stats_per_worker(dataframe=original_results.dataframe_numbered)
original_results.get_stats_per_audio(dataframe=original_results.dataframe_numbered)
