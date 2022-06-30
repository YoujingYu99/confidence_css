"""Extracts features from a all audio files.

This script extracts useful features for confidence assessment from all 
audio files and save to a csv.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from features import *


def extract_features_from_folders(home_dir, folder_path_list):
    """
    Extract features for all folders.
    :param folder_path_list: List of folder-level paths
    :return: Extract features and save to csvs
    """
    audio_path_list = audio_path_in_dir(folder_path_list)
    # Find folder number
    audio_folder_name = audio_path_list[0].split("/")[-2]
    for audio_path in audio_path_list:
        # Create csv name
        feature_csv_folder_path = os.path.join(
            home_dir, "data_sheets", "features", str(audio_folder_name)
        )
        features = SingleFileFeatureExtraction(
            home_dir=home_dir,
            audio_path=audio_path,
            feature_csv_folder_path=feature_csv_folder_path,
        )
        features.write_features_to_csv()


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# # Create folder path list
# folder_path_list = []
# for i in range(8):
#     folder_path = os.path.join(home_dir, "extracted_audio", str(i))
#     folder_path_list.append(folder_path)

folder_path_list = [os.path.join(home_dir, "extracted_audio", str(0))]
extract_features_from_folders(home_dir, folder_path_list)
