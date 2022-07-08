"""Extract audio segments from their parent audio files.

This script uses information on the questions from the csv file to extract
short audio segments from the parent audio files.
"""

from data_preparation_utils import *

folder_number = 0
dataframe_name = "confidence_dataframe_" + str(folder_number) + ".csv"

home_dir = os.path.join("/home", "yyu")
# Get datasheet path
csv_path = os.path.join(
    home_dir, "data_sheets", "confidence_dataframes", dataframe_name
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
