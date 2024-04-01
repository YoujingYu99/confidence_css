"""Use pre-trained Bert and HuBERT models on the audio and text for regression.
Extract the raw audio array, transcription and confidence score from the individual audio
classes. Then use this data to train the network for regression. Specify home_dir before running the script.
"""

from transformers import AutoFeatureExtractor, BertTokenizer
import gc
from ast import literal_eval
from model_utils import *

# Decide on whether to tokenize audios before training or use raw audio arrays.
vectorise = True
two_scores = False
test_absolute = True
ablate_text = True

# Load feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# home_dir is the location of folder
home_dir = ""
folder_path = os.path.join(home_dir, "data", "label_results")

# Path for crowdsourcing results
crowdsourcing_results_train_df_path = os.path.join(
    folder_path,
    "Cleaned_Results_Train.csv",
)
crowdsourcing_results_val_df_path = os.path.join(
    home_dir,
    "Cleaned_Results_eval.csv",
)
# This must be the same as the human label dataset!
crowdsourcing_results_test_df_path = os.path.join(
    home_dir,
    "Cleaned_Results_Test.csv",
)


print("start of application!")

audio_text_train_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_train_df_path,
    save_to_single_csv=False,
    augment_audio=True,
    two_scores=two_scores,
)


audio_text_test_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_test_df_path,
    save_to_single_csv=False,
    augment_audio=False,
    two_scores=two_scores,
)

audio_text_val_df = load_audio_text_and_score_from_crowdsourcing_results(
    home_dir,
    crowdsourcing_results_val_df_path,
    save_to_single_csv=False,
    augment_audio=False,
    two_scores=two_scores,
)

# Training parameters
epochs = 1500
LR = 1e-7
weight_decay = 1e-9
batch_size = 8
num_workers = 4
accum_iter = 4

if ablate_text:
    ## Text ablation: zero text tokens and use audio only
    model = CustomMultiModelSimplePooledText()
    train_audio_text_ablation(
        model,
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
        freeze="first_ele",
        ablation_type="text",
    )

    # # Reload from checkpoint for evaluation
    # checkpoint_path = os.path.join(
    #     home_dir,
    #     "model_checkpoints_ablation",
    #     "" + "_checkpoint.pt",
    # )
    # # load the last checkpoint with the best model
    # model.load_state_dict(torch.load(checkpoint_path), strict=False)

    # evaluate_audio_text_ablation(
    #     model,
    #     audio_feature_extractor,
    #     text_tokenizer,
    #     audio_text_test_df,
    #     batch_size,
    #     vectorise,
    #     test_absolute,
    #     type="text",
    #     model_name="",
    # )
else:
    ## Audio ablation: zero audio tokens and use text only
    model = CustomMultiModelSimplePooledAudio()
    train_audio_text_ablation(
        model,
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
        freeze="first_ele",
        ablation_type="text",
    )

    # # Reload from checkpoint for evaluation
    # checkpoint_path = os.path.join(
    #     home_dir,
    #     "model_checkpoints_ablation",
    #     "" + "_checkpoint.pt",
    # )

    # model.load_state_dict(torch.load(checkpoint_path), strict=False)

    # evaluate_audio_text_ablation(
    #     model,
    #     audio_feature_extractor,
    #     text_tokenizer,
    #     audio_text_test_df,
    #     batch_size,
    #     vectorise,
    #     test_absolute,
    #     type="audio",
    #     model_name="",
    # )
