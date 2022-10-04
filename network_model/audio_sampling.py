"""Clean up audios to be used in the used_audios folder.
"""
import os
import pandas as pd
import shutil

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")


# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned.csv",
)

extracted_audio_path = os.path.join(home_dir, "extracted_audios")
used_audio_path = os.path.join(home_dir, "used_audios")


used_audio_urls = pd.read_csv(crowdsourcing_results_df_path)["audio_url"].tolist()
print("start of application!")

for audio_url in used_audio_urls:

    folder_number = audio_url.split("/")[-2]
    segment_name = audio_url.split("/")[-1][:-4]
    old_audio_path = os.path.join(
        extracted_audio_path, str(folder_number), segment_name + ".mp3",
    )
    new_path = os.path.join(used_audio_path, str(folder_number), segment_name + ".mp3",)
    # Move to new destination
    shutil.move(old_audio_path, new_path)
