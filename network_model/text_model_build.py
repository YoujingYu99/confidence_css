"""Use pre-trained BERT model on the text for classification.
Extract the text and confidence score to a csv. Then from the csv train the
BERT model for confidence classification.
"""
import os
import pandas as pd
import torch
from model_utils import *
from models import *


torch.cuda.empty_cache()
# Load dataset
home_dir = os.path.join("/home", "yyu")
train_csv_dir = os.path.join(home_dir, "data_sheets", "train_new.csv")
df = pd.read_csv(train_csv_dir)
print(df.head())
df_train, df_val, df_test = np.split(
    df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
)
print(len(df_train), len(df_val), len(df_test))
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Decide on Epoch and model
EPOCHS = 5
text_model = BertClassifier()
LR = 1e-6

# Train model
train_text(text_model, df_train, df_val, LR, EPOCHS)
# evaluate_text(model, df_test)
