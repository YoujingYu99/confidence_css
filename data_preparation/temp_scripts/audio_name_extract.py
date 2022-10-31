"""Generate audio names to be uploaded.

This script generates a txt file with the names of the audio files to be used to give
Dr Lipani so as to avoid uploading of unnecessary audios."""


import os
import pandas as pd

home_dir = os.path.join("/home", "yyu")
folder_numbers = range(8)
output_path = os.path.join(home_dir, "data_sheets", "audio_names_file.txt")

filename_list = []
# Iterate over folders
for folder_number in folder_numbers:
    dataframe_name = "confidence_dataframe_" + str(folder_number) + ".csv"
    csv_path = os.path.join(
        home_dir, "data_sheets", "confidence_dataframes", dataframe_name
    )
    filenames = pd.read_csv(csv_path)["filename"]
    for filename in filenames.tolist():
        # Sub folder list
        top_folder_name = filename.split("/")[-4]
        folder_name = filename.split("/")[-3]
        subfolder_name = filename.split("/")[-2]
        audio_name = filename.split("/")[-1][:-5]
        clean_name = (
            str(top_folder_name)
            + "/"
            + str(folder_name)
            + "/"
            + str(subfolder_name)
            + "/"
            + str(audio_name)
            + ".ogg"
        )
        filename_list.append(clean_name)

# Remove duplicates
uniq_list = list(dict.fromkeys(filename_list))
with open(output_path, "w+") as myfile:
    for line in uniq_list:
        myfile.write(str(line) + "\n")
