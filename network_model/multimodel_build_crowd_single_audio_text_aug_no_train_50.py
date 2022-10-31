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

# # Path for crowdsourcing results
# crowdsourcing_results_train_df_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
#     "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_train4.csv",
# )
# crowdsourcing_results_val_df_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
#     "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_val4.csv",
# )
# crowdsourcing_results_test_df_path = os.path.join(
#     home_dir,
#     "data_sheets",
#     "crowdsourcing_results",
#     "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_test4.csv",
# )

# Training parameters
epochs = 3
LR = 5e-6
weight_decay = 1e-8
batch_size = 16
num_workers = 4
accum_iter = 4


print("start of application!")
# split_to_train_val_test_many(home_dir)

val_loss_list = []
val_acc_list = []
# for i in range(50):
#     crowdsourcing_results_val_df_path = os.path.join(
#         home_dir,
#         "data_sheets",
#         "crowdsourcing_results",
#         "split_tests",
#         "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_val"
#         + str(i)
#         + ".csv",
#     )
#
#     # crowdsourcing_results_val_df_path = os.path.join(
#     #     home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
#     # )
#
#     audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
#         home_dir,
#         crowdsourcing_results_val_df_path,
#         save_to_single_csv=False,
#         augment_audio=False,
#         two_scores=two_scores,
#     )
#     multimodel = CustomMultiModelSimplePooled()
#
#     # Train model
#     val_loss, val_acc = no_train_audio_text_many(
#         multimodel,
#         audio_feature_extractor,
#         text_tokenizer,
#         audio_text_val_df,
#         LR,
#         weight_decay,
#         epochs,
#         batch_size,
#         num_workers,
#         accum_iter,
#         vectorise,
#         test_absolute,
#     )
#     val_loss_list.append(val_loss)
#     val_acc_list.append(val_acc)
#
#
# list_of_tuples_loss_acc = list(
#         zip(val_loss_list, val_acc_list,)
#     )
# loss_acc_df = pd.DataFrame(
#     list_of_tuples_loss_acc,
#     columns=["Val Loss", "Val Acc",],
# )
#
# loss_acc_df.to_csv(
#     os.path.join(
#         "/home", "yyu", "plots", "training_csv", "random_run_loss_acc.csv"
#     ),
#     index=False,
# )



crowdsourcing_results_val_df_path = os.path.join(
    home_dir,
    "data_sheets",
    "crowdsourcing_results",
    "split_tests",
    "Batch_4799159_batch_results_complete_reject_filtered_numbered_cleaned_val.csv",
)

# crowdsourcing_results_val_df_path = os.path.join(
#     home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
# )

audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_val_df_path,
    save_to_single_csv=False,
    augment_audio=False,
    two_scores=two_scores,
)
multimodel = CustomMultiModelSimplePooled()


# Do not train model
val_loss, val_acc = no_train_audio_text_many(
    multimodel,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_val_df,
    LR,
    weight_decay,
    epochs,
    batch_size,
    num_workers,
    accum_iter,
    vectorise,
    test_absolute,
)

print(val_acc)

