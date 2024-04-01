"""Add benchmark example audios to the batch.

This scripts add benchmark example audios to the batch csv and save to new
csv. This outputs the total batch that we are going to use.
"""

import pandas as pd
import os
import random
import math

home_dir = ""
benchmark_url_df = pd.read_csv(
    os.path.join(home_dir, "data_sheets", "sw3_urls", "Benchmark_Samples.csv")
)


## Concatenate all input dataframes; input + num is the dataframe uploaded to Amazon MTurk
pd_list = []
for i in range(7):
    input_dataframe = pd.read_csv(
        os.path.join(home_dir, "data_sheets", "sw3_urls", "input_" + str(i) + ".csv")
    )
    pd_list.append(input_dataframe)
total_df = pd.concat(pd_list, ignore_index=True)

# Size of total dataframes; this is also the total number of audio clips to be rated by workers.
total_num_audios = total_df.size


def repeat_benchmark_df(total_num_audios, original_df):
    """
    Repeat the benchmark dataframe so it has roughly the same number of rows as the total_df.
    :param total_num_audios: Total number of audios in batch.
    :param original_df: The original marked benchmark dataframe.
    :return: The benchmarked dataframe repeated to have more rows than number of audios.
    """

    number_test_audios = len(original_df)
    num_times_test_audios_repeated = (
        math.ceil(total_num_audios / number_test_audios) + 1
    )
    # Make a copy of test dataframe to work on
    benchmark_url_df_new = benchmark_url_df.copy()
    # Repeat the test dataframe so the number of rows matches the total number of samples
    for i in range(num_times_test_audios_repeated):
        benchmark_url_df_new = pd.concat(
            [benchmark_url_df_new, benchmark_url_df], ignore_index=True
        )
    # Shuffle rows
    benchmark_url_df_new = benchmark_url_df_new.sample(frac=1).reset_index(drop=True)
    return benchmark_url_df_new


benchmark_url_df_new = repeat_benchmark_df(total_num_audios, benchmark_url_df)

# Get number of HITs
num_of_HITs = total_df.shape[0]
# Choose the num_of_HITs benchmarked urls.
benchmark_urls_1 = benchmark_url_df_new["audio_url"][:num_of_HITs]
# Shuffle again and choose the num_of_HITs benchmarked urls.
benchmark_url_df_shuffled = benchmark_url_df_new.sample(frac=1).reset_index(drop=True)
benchmark_urls_2 = benchmark_url_df_shuffled["audio_url"][:num_of_HITs]
# Assign values to a new column in the total df
total_df["audio_url_11"] = benchmark_urls_1.values
total_df["audio_url_12"] = benchmark_urls_2.values
# Get column names
column_name_list = list(total_df.columns)
# shuffle each row and save to a list of lists
total_list = []
for index, row in total_df.iterrows():
    row_list = row.tolist()
    random.shuffle(row_list)
    total_list.append(row_list)
new_df = pd.DataFrame(total_list, columns=column_name_list)

# Save to new csv; input_total_bench is the total plus benchmarked csv to be uploaded to Amazon MTurk
csv_output_path = os.path.join(
    home_dir, "data_sheets", "sw3_urls", "input_total_bench.csv"
)
new_df.to_csv(csv_output_path, index=False)
