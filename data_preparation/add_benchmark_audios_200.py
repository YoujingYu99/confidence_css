"""Add benchmark example audios to the batch.

This scripts add benchmark example audios to the batch csv and sabe to new
csv
"""


import pandas as pd
import os
import random

home_dir = os.path.join("/home", "yyu")
benchmark_url_df = pd.read_csv(
    os.path.join(
        home_dir, "data_sheets", "sw3_urls", "samples_benchmark_200_marked.csv"
    )
)

## Concatenate input dataframes
# list of dataframes
pd_list = []
for i in range(7):
    input_dataframe = pd.read_csv(
        os.path.join(home_dir, "data_sheets", "sw3_urls", "input_" + str(i) + ".csv")
    )
    pd_list.append(input_dataframe)
total_df = pd.concat(pd_list, ignore_index=True)

benchmark_url_df_new = benchmark_url_df.copy()

# Extend the dataset 40 times so the number of rows matches the total number of samples
for i in range(40):
    benchmark_url_df_new = pd.concat(
        [benchmark_url_df_new, benchmark_url_df], ignore_index=True
    )
# Shuffle rows
benchmark_url_df_new = benchmark_url_df_new.sample(frac=1).reset_index(drop=True)

# Get number of HITs
num_of_HITs = total_df.shape[0]
# Choose 20 urls
benchmark_urls_1 = benchmark_url_df_new["audio_url"][:num_of_HITs]
# Shuffle again
benchmark_url_df_shuffled = benchmark_url_df_new.sample(frac=1).reset_index(drop=True)
benchmark_urls_2 = benchmark_url_df_shuffled["audio_url"][:num_of_HITs]
# Assign values to a new column in the dataframe
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

# Save to new csv
csv_output_path = os.path.join(
    home_dir, "data_sheets", "sw3_urls", "input_total_bench.csv"
)
new_df.to_csv(csv_output_path, index=False)
