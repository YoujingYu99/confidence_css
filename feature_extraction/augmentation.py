"""Augment audios for training. NOT IN USE."""
import nlpaug.augmenter.audio as naa
import os
import numpy as np
import pandas as pd
import ast


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
featuers_folder_path_dir = os.path.join(home_dir, "data_sheets", "features")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_renamed_soft.csv",
)


audio_text_all_df = pd.read_csv(
    os.path.join(home_dir, "data_sheets", "audio_text_crowd_all_model.csv")
)
print("number of rows of all models", audio_text_all_df.shape[0])

audio_text_aug_df = audio_text_all_df.copy()

sr = 16000


def augment_audio(audio_array):
    """
    Augment audio into new arrays.
    :param audio_array: Original audio array.
    :return: augmented audio array.
    """
    # Loudness
    aug = naa.LoudnessAug()
    augmented_data = aug.augment(audio_array)
    # Noise
    aug = naa.NoiseAug()
    augmented_data = aug.augment(augmented_data)
    # Pitch
    aug = naa.PitchAug(sampling_rate=sr, factor=(2, 3))
    augmented_data = aug.augment(augmented_data)
    return augmented_data


# Augment audio
for index, row in audio_text_aug_df.iterrows():
    original_audio = row["audio_array"]
    original_audio = ast.literal_eval(original_audio)
    new_audio = augment_audio(np.array(original_audio))
    row["audio_array"] = new_audio

save_path = os.path.join(
    home_dir, "data_sheets", "audio_text_crowd_all_model_audioaug.csv"
)
audio_text_aug_df.to_csv(save_path, index=False)
