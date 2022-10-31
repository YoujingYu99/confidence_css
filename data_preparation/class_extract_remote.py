"""Extract questions from json transcripts.

This script finds the questions from json transcripts and gather all
information in a csv file.
"""
from data_preparation_utils import *

folder_number = 0

home_dir = os.path.join("/home", "yyu")
file_dir = os.path.join(
    home_dir,
    "data",
    "Spotify-Podcasts",
    "podcasts-no-audio-13GB",
    "decompressed-transcripts",
)

app_dir = os.path.join(file_dir, str(folder_number))

# Extract all to one dataframe
extract_complete_dataframe(home_dir, folder_number, folder_path_list=[app_dir])
