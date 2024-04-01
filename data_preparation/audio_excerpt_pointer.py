"""Extract audio segments from their parent audio files.

Specify home_dir, audio_dir and extracted_dir on your local computer. Run this script to generate the audio excerpts of the dataset.
"""

from data_preparation_utils import *

# project dir
home_dir = ""
# Audio data file path
audio_dir = ""
# Output path
extracted_dir = ""


# Extract audio files from folders 0 to 7. Folder 7 audios correspond to samples_benchmark.
folder_number_list = range(8)
for folder_number in folder_number_list:

    # Get datasheet path
    csv_path = os.path.join(
        home_dir,
        "data",
        "confidence_dataframes",
        "confidence_dataframe_total_remove.csv",
    )
    audio_path = os.path.join(
        audio_dir,
        "Spotify-Podcasts",
        "podcasts-audio-only-2TB",
        "podcasts-audio",
        str(folder_number),
    )

    excerpt_output_path = os.path.join(extracted_dir, str(folder_number))

    # Extract audios
    extract_all_audios(csv_path, [audio_path], excerpt_output_path)
