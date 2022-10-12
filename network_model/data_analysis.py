"""Analysis of the score distribution, the text token length and the audio token length
on the original dataset.
"""
import os
import pandas as pd
from transformers import AutoFeatureExtractor, BertTokenizer
from model_utils import (
    load_text_and_score_from_crowdsourcing_results,
    load_audio_and_score_from_crowdsourcing_results,
    plot_histogram_of_scores,
)

# Decide whether to save the concatenated file to a single csv
save_to_single_csv = False
# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
# Load feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
featuers_folder_path_dir = os.path.join(home_dir, "data_sheets", "features")

# # Path for crowdsourcing results
# crowdsourcing_results_df_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
# "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_train.csv",
# )

crowdsourcing_results_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
"test_crowd.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
# audio_text_df_train = load_audio_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_df_path,
#     save_to_single_csv,
#     augment_audio=False,
#     two_scores=False,
# )
#
# audio_df_train = load_audio_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_df_path,
#     save_to_single_csv,
#     augment_audio=False,
#     two_scores=False,
# )

# text_df_train = load_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_df_path,
#     save_to_single_csv,
#     augment_text=False,
#     two_scores=False,
# )

# ## Analysis of score distribution
# train_scores_list = audio_df_train["score"].tolist()
# train_scores_origin_list = [i + 2.5 for i in train_scores_list]
# plot_histogram_of_scores(
#     home_dir,
#     train_scores_origin_list,
#     num_bins=5,
#     plot_name="Score Distribution",
#     x_label="Scores",
# )


# # Analysis of audio token length
audio_df_train = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv,augment_audio=False,
    two_scores=False,
)
#
audio_series = audio_df_train["audio_array"]
audio_tensor_length_list = []
print("Start tokenisation")
for audio in audio_series:
    extracted_tensor = audio_feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt",
    )
    print(extracted_tensor.input_values.size())
    audio_tensor_length_list.append(list(extracted_tensor.input_values.size())[1])

# df = pd.DataFrame(audio_tensor_length_list, columns=["Audio Token Length"])
# df.to_csv(os.path.join(home_dir, "plots", "audio_token_length.csv"))
# plot_histogram_of_scores(home_dir, audio_tensor_length_list, num_bins=30, plot_name="Audio Token Length", x_label="Token Length")



# # Analysis of text token length
# text_df_train = load_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_df_path,
#     save_to_single_csv,
#     augment_text=False,
#     two_scores=False,
# )
# texts = text_df_train["sentence"]
# text_tensor_length_list = []
# for text in texts:
#     extracted_tensor = text_tokenizer(text, return_tensors="pt")
#     text_tensor_length_list.append(len(extracted_tensor.input_ids[0]))
#
# print("Start plotting!")
# df = pd.DataFrame(text_tensor_length_list, columns=["Text Token Length"])
# df.to_csv(os.path.join(home_dir, "plots", "text_token_length.csv"))
# plot_histogram_of_scores(
#     home_dir,
#     text_tensor_length_list,
#     num_bins=10,
#     plot_name="Text Token Length",
#     x_label="Token Length",
# )

# print(
#     "number of audios with more than 120 tensors",
#     sum(1 for i in text_tensor_length_list if i > 120),
# )
