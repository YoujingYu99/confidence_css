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
two_scores = True
test_absolute = True

# Upsample and augment if train data not ready
upsample_train = False
augment_audio_train = False
augment_text_train = False
concat_final_train_df = False
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


# # Only uncomment if need to reupsample data / augment again.
# upsample_audio_text_augment(
#     upsample_train,
#     augment_audio_train,
#     augment_text_train,
#     concat_final_train_df,
#     home_dir,
#     crowdsourcing_results_train_df_path,
#     two_scores,
# )


def convert_str_to_list(input_str):
    return json.loads(input_str)


# Load train
mylist = []
for chunk in pd.read_csv(
    os.path.join(
        "/home", "yyu", "data_sheets", "audio_text_upsampled_augmented_total.csv",
    ),
    chunksize=100,
    usecols=["audio_array", "sentence", "score"],
    converters={"audio_array": literal_eval, "score": literal_eval},
):
    # chunk["audio_array"] = chunk[
    #     "audio_array"].map(convert_str_to_list)
    mylist.append(chunk)


audio_text_train_df = pd.concat(mylist, axis=0)
del mylist
gc.collect()


audio_text_val_df = pd.read_csv(
    os.path.join(home_dir, "data_sheets", "audio_text_crowd_val.csv"),
    usecols=["audio_array", "sentence", "score"],
    converters={"audio_array": literal_eval, "score": literal_eval},
)

# audio_text_val_df["audio_array"] = audio_text_val_df["audio_array"].map(convert_str_to_list)
audio_text_test_df = pd.read_csv(
    os.path.join(home_dir, "data_sheets", "audio_text_crowd_test.csv"),
    usecols=["audio_array", "sentence", "score"],
    converters={"audio_array": literal_eval, "score": literal_eval},
)

# audio_text_test_df["audio_array"] = audio_text_test_df["audio_array"].map(convert_str_to_list)


# Training parameters
epochs = 1500
LR = 5e-6
weight_decay = 1e-6
batch_size = 8
num_workers = 4
accum_iter = 2

# Initialise audio model
# audio_model = HubertClassifier()
multimodel = CustomMultiModelSimplePooled()

print("Start training!")
# Train model
train_audio_text(
    multimodel,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_train_df,
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

evaluate_audio_text(
    multimodel,
    audio_feature_extractor,
    text_tokenizer,
    audio_text_test_df,
    batch_size,
    vectorise,
    test_absolute,
)
