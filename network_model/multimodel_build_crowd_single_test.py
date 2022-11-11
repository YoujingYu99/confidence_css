"""Use pre-trained Bert and HuBERT models on the audio and text for regression.
Extract the raw audio array, transcription and confidence score from the individual audio
classes. Then use this data to train the network for regression.
"""
from transformers import AutoFeatureExtractor, BertTokenizer
import gc
from ast import literal_eval
from model_utils import *

# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
two_scores = False
test_absolute = True

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
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)


print("start of application!")

# audio_text_train_df = load_audio_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_train_df_path,
#     save_to_single_csv=False,
#     augment_audio=True,
#     two_scores=two_scores,
# )
#
#
# audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
#     home_dir,
#     crowdsourcing_results_val_df_path,
#     save_to_single_csv=False,
#     augment_audio=False,
#     two_scores=two_scores,
# )

audio_text_test_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_test_df_path,
    save_to_single_csv=False,
    augment_audio=False,
    two_scores=two_scores,
)

# Training parameters
epochs = 1500
# LR = 5e-6
weight_decay = 1e-9
batch_size = 8
num_workers = 4
accum_iter = 4

# multimodel = CustomMultiModelSimplePooled()
# models = [CustomMultiModelSimplePooled(), CustomMultiModelSimplePooledThreeLayers()]
model = CustomMultiModelSimplePooled()


print("Start training!")
#
# validation_pairs = [[1e-6, "first_ele"], [5e-6, "first_ten"]]
#
# for validation_pair in validation_pairs:
#     LR = validation_pair[0]
#     freezing_mode = validation_pair[1]
#
#     train_audio_text(
#         model,
#         audio_feature_extractor,
#         text_tokenizer,
#         audio_text_train_df,
#         audio_text_val_df,
#         LR,
#         weight_decay,
#         epochs,
#         batch_size,
#         num_workers,
#         accum_iter,
#         vectorise,
#         test_absolute,
#         freeze=freezing_mode,
#     )
#
#     evaluate_audio_text(
#         model,
#         audio_feature_extractor,
#         text_tokenizer,
#         audio_text_test_df,
#         batch_size,
#         vectorise,
#         test_absolute,
#     )

model_name = "upsample_augment_three_run_one_first_ten_1e-07_"

checkpoint_path = os.path.join(
    "/home", "yyu", "model_checkpoints", model_name + "_checkpoint.pt"
)
# load the last checkpoint with the best model
model.load_state_dict(torch.load(checkpoint_path), strict=False)

type = "text"
evaluate_audio_text_ablation(
    model,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_test_df,
    batch_size,
    vectorise,
    test_absolute,
    type,
)
