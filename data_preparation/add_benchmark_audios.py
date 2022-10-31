"""Add benchmark example audios to the batch.

This scripts add benchmark example audios to the batch csv and save to new
csv.
"""


import pandas as pd
import os
import random

home_dir = os.path.join("/home", "yyu")
folder_numbers = [0]


benchmark_url_df = pd.read_csv(
    os.path.join(
        home_dir, "data_sheets", "sw3_urls", "Benchmark_Samples.csv"
    )
)

for folder_number in folder_numbers:
    datasheet_name = "input_" + str(folder_number) + ".csv"
    original_df = pd.read_csv(
        os.path.join(home_dir, "data_sheets", "sw3_urls", datasheet_name)
    )
    num_of_HITs = original_df.shape[0]
    # Choose 20 urls
    benchmark_urls_1 = benchmark_url_df["audio_url"][:num_of_HITs]
    benchmark_urls_2 = benchmark_url_df["audio_url"][num_of_HITs : 2 * num_of_HITs]
    # Assign values to a new column in the dataframe
    original_df["audio_url_11"] = benchmark_urls_1.values
    original_df["audio_url_12"] = benchmark_urls_2.values
    # Get column names
    column_name_list = list(original_df.columns)
    # shuffle each row and save to a list of lists
    total_list = []
    for index, row in original_df.iterrows():
        row_list = row.tolist()
        random.shuffle(row_list)
        total_list.append(row_list)
    new_df = pd.DataFrame(total_list, columns=column_name_list)
    # Save to new csv
    datasheet_name_benchmark = "input_" + str(folder_number) + "_bench" + ".csv"
    csv_output_path = os.path.join(
        home_dir, "data_sheets", "sw3_urls", datasheet_name_benchmark
    )
    new_df.to_csv(csv_output_path, index=False)
