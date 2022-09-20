"""Use pre-trained Bert and HuBERT models on the audio and text for regression.
Extract the raw audio array, transcription and confidence score from the individual audio
classes. Then use this data to train the network for regression.
"""
from transformers import AutoFeatureExtractor, BertTokenizer
from model_utils import *

# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
two_scores = True
test_absolute = True

# Upsample and augment if train data not ready
upsample_train = False
augment_audio_train = True
augment_text_train = True
concat_final_train_df = True
# Load feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_train_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_train.csv",
)
crowdsourcing_results_val_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_val.csv",
)
crowdsourcing_results_test_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test.csv",
)


print("start of application!")

upsample_audio_text_augment(
    upsample_train,
    augment_audio_train,
    augment_text_train,
    concat_final_train_df,
    home_dir,
    crowdsourcing_results_train_df_path,
    two_scores,
)

# Load train
mylist = []
for chunk in pd.read_csv(
    os.path.join(
        "/home", "yyu", "data_sheets", "audio_text_upsampled_augmented_total",
    ),
    chunksize=1000,
):
    mylist.append(chunk)

audio_text_train_df = pd.concat(mylist, axis=0)
del mylist

# audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_val_df_path,
#     save_to_single_csv=False,
#     augment_audio=False,
#     two_scores=two_scores,
# )
#
# audio_text_test_df = load_audio_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_test_df_path,
#     save_to_single_csv=False,
#     augment_audio=False,
#     two_scores=two_scores,
# )
#
#
# # Training parameters
# epochs = 1500
# LR = 5e-6
# weight_decay = 1e-6
# batch_size = 8
# num_workers = 4
# accum_iter = 2
#
# # Initialise audio model
# # audio_model = HubertClassifier()
# multimodel = CustomMultiModelSimplePooled()
#
# # Train model
# train_audio_text(
#     multimodel,
#     audio_feature_extractor,
#     text_tokenizer,
#     audio_text_train_df,
#     audio_text_val_df,
#     LR,
#     weight_decay,
#     epochs,
#     batch_size,
#     num_workers,
#     accum_iter,
#     vectorise,
#     test_absolute,
# )
#
# evaluate_audio_text(
#     multimodel,
#     audio_feature_extractor,
#     text_tokenizer,
#     audio_text_test_df,
#     batch_size,
#     vectorise,
#     test_absolute,
# )
