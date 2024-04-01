"""Extract audio segments from their parent audio files.

Run this script to generate the audio excerpts of the dataset.
"""

from data_preparation_utils import *


home_dir = os.path.join("/home", "youjing", "PersonalProjects", "confidence_css")
# Get datasheet path
folder_path = os.path.join(home_dir, "data", "confidence_dataframes")


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
