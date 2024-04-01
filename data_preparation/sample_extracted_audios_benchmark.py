"""Set a number of samples to be extracted (200) from folder 7 and move them to
the chosen folder. These are used for benchmarked samples.
"""

import os
import random
import shutil


folder_numbers = [7]
no_of_files = 200

home_dir = ""

# Get datasheet path
old_path = os.path.join(home_dir, "extracted_audio", "extracted_audio_total")
new_path = os.path.join(
    home_dir, "extracted_audio", "extracted_audio_samples_benchmark_200"
)


for i in folder_numbers:
    source = os.path.join(old_path, str(i))
    dest = new_path
    files = os.listdir(source)

    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(source, file_name), dest)
    print("Done 1 folder!")
