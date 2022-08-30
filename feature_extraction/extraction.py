"""Extracts features from a all audio files.

This script extracts useful features for confidence assessment from all 
audio files and save to a csv. The user can choose whether to extract just
the audio arrays or all features.
"""
import warnings
import time

warnings.simplefilter(action="ignore", category=FutureWarning)

from features import *

# Set target sampling rate
target_sampling_rate = 16000
folder_number = 2


def extract_features_from_folders(home_dir, folder_path_list, type):
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
        audio_array_csv_folder_path = os.path.join(
            home_dir, "data_sheets", "features_audio_array", str(audio_folder_name)
        )
        feature_csv_path = os.path.join(
            feature_csv_folder_path, str(audio_path.split("/")[-1][:-4]) + ".csv"
        )
        audio_array_csv_path = os.path.join(
            audio_array_csv_folder_path,
            str(audio_path.split("/")[-1][:-4]) + "_audio_only" + ".csv",
        )
        print(feature_csv_path)
        if type == "audio_array":
            # Skip files that have already been extracted
            if not os.path.isfile(audio_array_csv_path):
                try:
                    features = SingleFileFeatureExtraction(
                        home_dir=home_dir,
                        audio_path=audio_path,
                        feature_csv_folder_path=feature_csv_folder_path,
                        audio_array_csv_folder_path=audio_array_csv_folder_path,
                        target_sampling_rate=target_sampling_rate,
                    )
                    # features.write_features_to_csv()
                    features.write_audio_array_to_csv()
                except:
                    continue
        elif type == "all_features":
            # Skip files that have already been extracted
            if not os.path.isfile(feature_csv_path):
                try:
                    features = SingleFileFeatureExtraction(
                        home_dir=home_dir,
                        audio_path=audio_path,
                        feature_csv_folder_path=feature_csv_folder_path,
                        audio_array_csv_folder_path=audio_array_csv_folder_path,
                        target_sampling_rate=target_sampling_rate,
                    )
                    features.write_features_to_csv()
                except:
                    continue


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

folder_path_list = [os.path.join(home_dir, "extracted_audios", str(folder_number))]
extract_features_from_folders(home_dir, folder_path_list, "all_features")
