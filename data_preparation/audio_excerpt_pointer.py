"""Extract audio segments from their parent audio files.

This script uses information in the csv file Cleaned_Results to generate an audio pointer.
"""

from data_preparation_utils import *


folder_number = 7
dataframe_name = "confidence_dataframe_" + str(folder_number) + ".csv"

home_dir = os.path.join("/home", "youjing", "PersonalProjects", "confidence_css")
# Get datasheet path
folder_path = os.path.join(home_dir, "data", "confidence_dataframes")


# ## Step 1: concatenate all the confidence data frames
# all_files = os.listdir(folder_path)

# # Filter out non-CSV files
# csv_files = [f for f in all_files if f.endswith(".csv")]


# # Create a list to hold the dataframes
# df_list = []

# for csv in csv_files:
#     file_path = os.path.join(folder_path, csv)
#     try:
#         # Try reading the file using default UTF-8 encoding
#         df = pd.read_csv(file_path)
#         df_list.append(df)
#     except UnicodeDecodeError:
#         try:
#             # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
#             df = pd.read_csv(file_path, sep="\t", encoding="utf-16")
#             df_list.append(df)
#         except Exception as e:
#             print(f"Could not read file {csv} because of error: {e}")
#     except Exception as e:
#         print(f"Could not read file {csv} because of error: {e}")

# # Concatenate all data into one DataFrame
# big_df = pd.concat(df_list, ignore_index=True)
# print("saving csvs")
# # Save the final result to a new CSV file
# big_df.to_csv(os.path.join(folder_path, "combined_file.csv"), index=False)


# ## Step 2: compare with Cleaned_Results
# tot_conf_dataframe_path = os.path.join(folder_path, "combined_file.csv")
# tot_conf_df = pd.read_csv(tot_conf_dataframe_path)
# cleaned_results_dataframe_path = os.path.join(
#     home_dir, "data", "Label_Results", "Cleaned_Results.csv"
# )
# cleaned_results_df = pd.read_csv(cleaned_results_dataframe_path)


# def pair_columns(df, col1, col2):
#     return df[col1] + df[col2]


# def paired_mask(df1, df2, col1, col2):
#     return pair_columns(df1, col1, col2).isin(pair_columns(df2, col1, col2))


# def combined_file_filter(tot_conf_df, cleaned_results_df):

#     # List of filenames
#     audio_urls = cleaned_results_df.audio_url.astype(str)
#     new_file_name_list = []
#     new_start_time_list = []
#     for audio_url in audio_urls:
#         start_time = audio_url.split("_")[-1][:-4]
#         json_name = audio_url.split("_")[-2]
#         show_name = "show" + "_" + audio_url.split("_")[-3]
#         subfolder_name = audio_url.split("/")[-1][0]
#         folder_name = audio_url.split("/")[-2]

#         new_file_name = (
#             "/home/yyu/data/Spotify-Podcasts/podcasts-no-audio-13GB/decompressed-transcripts/"
#             + str(folder_name)
#             + "/"
#             + str(subfolder_name)
#             + "/"
#             + show_name
#             + "/"
#             + json_name
#             + ".json"
#         )
#         # change start time to the format in confidence_dataframe
#         if start_time[-1] == "0":
#             new_start_time = str(start_time)[:-2] + "s"
#         else:
#             new_start_time = str(start_time) + "00" + "s"
#         new_file_name_list.append(new_file_name)
#         new_start_time_list.append(new_start_time)
#         filtered_audio_name_pd = pd.DataFrame(
#             list(zip(new_start_time_list, new_file_name_list)),
#             columns=["start_time", "filename"],
#         )

#         tot_conf_df_filtered = tot_conf_df.loc[
#             paired_mask(tot_conf_df, filtered_audio_name_pd, "filename", "start_time")
#         ]
#         tot_conf_df_filtered_single = tot_conf_df_filtered.drop_duplicates()
#         # save to csv
#         tot_conf_df_filtered_single.to_csv(
#             os.path.join(folder_path, "combined_file_filtered.csv"), index=False
#         )


# # combined_file_filter(tot_conf_df, cleaned_results_df)
# tot_conf_df_filtered = pd.read_csv(
#     os.path.join(folder_path, "combined_file_filtered.csv")
# )

# # Test samples benchmark
# audio_urls = cleaned_results_df.audio_url.astype(str)


# new_file_name_list = []
# new_start_time_list = []
# for audio_url in audio_urls:
#     start_time = audio_url.split("_")[-1][:-4]
#     json_name = audio_url.split("_")[-2]
#     show_name = "show" + "_" + audio_url.split("_")[-3]
#     subfolder_name = audio_url.split("/")[-1][0]
#     folder_name = audio_url.split("/")[-2]

#     # change start time to the format in confidence_dataframe
#     if start_time[-1] == "0":
#         new_start_time = str(start_time)[:-2] + "s"
#     else:
#         new_start_time = str(start_time) + "00" + "s"
#     if folder_name == "samples_benchmark_200":
#         # sample benchmark files are all generated from folder 7
#         new_file_name = (
#             "/home/yyu/data/Spotify-Podcasts/podcasts-no-audio-13GB/decompressed-transcripts/"
#             + "7"
#             + "/"
#             + str(subfolder_name)
#             + "/"
#             + show_name
#             + "/"
#             + json_name
#             + ".json"
#         )
#         new_file_name_list.append(new_file_name)
#         new_start_time_list.append(new_start_time)

#     sample_benchmark_audio_name_pd = pd.DataFrame(
#         list(zip(new_start_time_list, new_file_name_list)),
#         columns=["start_time", "filename"],
#     )

#     # See if something matches in the confidence dataset total.
#     sample_benchmark_tot_conf_df_filtered = tot_conf_df.loc[
#         paired_mask(
#             tot_conf_df, sample_benchmark_audio_name_pd, "filename", "start_time"
#         )
#     ]
#     sample_benchmark_tot_conf_df_filtered = (
#         sample_benchmark_tot_conf_df_filtered.drop_duplicates()
#     )
#     # save to csv
#     sample_benchmark_tot_conf_df_filtered.to_csv(
#         os.path.join(folder_path, "sample_benchmark_filtered.csv"), index=False
#     )

# # Combine confidence dataframes and samples_benchmark
# df_list = []
# combined_file_filtered_df = pd.read_csv(
#     os.path.join(folder_path, "combined_file_filtered.csv")
# )
# df_list.append(combined_file_filtered_df)
# sample_benchmark_filtered_df = pd.read_csv(
#     os.path.join(folder_path, "sample_benchmark_filtered.csv")
# )
# df_list.append(sample_benchmark_filtered_df)


# # Concatenate all data into one DataFrame
# big_df = pd.concat(df_list, ignore_index=True)
# big_df.to_csv(os.path.join(folder_path, "confidence_dataframe_total.csv"), index=False)


# Step 3: extract audios from this filtered confidence dataframe
# Extract audio files from folders 0 to 7. Folder 7 audios correspond to samples_benchmark.
folder_number_list = range(8)
for folder_number in folder_number_list:

    # Get datasheet path
    csv_path = os.path.join(
        home_dir, "data", "confidence_dataframes", "combined_file_filtered.csv"
    )
    # Audio data file path
    fileDir = os.path.join(home_dir, "data")
    audio_path = os.path.join(
        fileDir,
        "Spotify-Podcasts",
        "podcasts-audio-only-2TB",
        "podcasts-audio",
        str(folder_number),
    )

    # Output path
    excerpt_output_path = os.path.join(home_dir, "extracted_audio", str(folder_number))

    # Extract audios
    extract_all_audios(csv_path, [audio_path], excerpt_output_path)
