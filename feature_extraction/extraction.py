"""Extracts features from a all audio files.

This script extracts useful features for confidence assessment from all 
audio files and save to a csv.
"""
from features import *

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Create folder path list
folder_path_list = []
for i in range(8):
    folder_path = os.path.join(home_dir, "extracted_audio", str(i))
    folder_path_list.append(folder_path)

extract_features_from_folders(home_dir, folder_path_list)
