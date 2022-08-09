"""Set a number of samples to be extracted from each folder and move them to
the chosen folder.
"""
import os
import random
import shutil

# folder_numbers = [0]
folder_numbers = range(7)
no_of_files = 1100

home_dir = os.path.join("/home", "yyu")
# Get datasheet path
old_path = os.path.join(home_dir, "extracted_audio")
new_path = os.path.join(home_dir, "extracted_audio_samples")

for i in folder_numbers:
    source = os.path.join(old_path, str(i))
    dest = os.path.join(new_path, str(i))
    files = os.listdir(source)

    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(source, file_name), dest)
    print("Done 1 folder!")
