"""Set a number of samples to be extracted from each folder and move them to
the chosen folder.
"""

import os
import random
import shutil
import boto3
import pandas as pd


folder_numbers = [7]
# folder_numbers = range(8)
no_of_files = 200
#
home_dir = os.path.join("/home", "yyu")

# Get datasheet path
old_path = os.path.join(home_dir, "extracted_audio", "extracted_audio_total")
new_path = os.path.join(
    home_dir, "extracted_audio", "extracted_audio_samples_benchmark_200"
)
existed_path = os.path.join(
    home_dir, "extracted_audio", "extracted_audio_samples_benchmark_20"
)

for i in folder_numbers:
    source = os.path.join(old_path, str(i))
    dest = new_path
    files = os.listdir(source)

    for file_name in random.sample(files, no_of_files):
        # Only move if file does not exist in directory
        if not os.path.isfile(os.path.join(existed_path, file_name) + ".mp3"):
            shutil.move(os.path.join(source, file_name), dest)
    print("Done 1 folder!")
