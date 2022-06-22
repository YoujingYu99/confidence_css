"""Extracts features from a all audio files.

This script extracts useful features for confidence assessment from all 
audio files and save to a csv.
"""
from features import *

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
folder_path_list = [os.path.join(home_dir, "extracted_audio", "1")]

audio_path_list = audio_path_in_dir(folder_path_list)
for audio_path in audio_path_list:
    # Create csv name
    feature_csv_folder_path = os.path.join(home_dir, "data_sheets", "features", "1")
    features = SingleFileFeatureExtraction(
        home_dir=home_dir,
        audio_path=audio_path,
        feature_csv_folder_path=feature_csv_folder_path,
    )
    features.write_features_to_csv()
