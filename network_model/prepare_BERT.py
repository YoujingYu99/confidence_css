import os
import pandas as pd
import random


home_dir = os.path.join("/home", "yyu")
train_csv_dir = os.path.join(home_dir, "data_sheets", "train.csv")
train_bert_csv_dir = os.path.join(home_dir, "data_sheets", "train_bert.csv")


# Currently use dummy dataset for model testing
train_df = pd.read_csv(train_csv_dir)
train_df.columns = ["score", "sentence"]
num_data = train_df.shape[0]
# Readjust the training dataset to have labels from 1 to 10.
confidence_values = range(1, 10, 1)

new_confidence = []
for i in range(num_data):
    new_confidence.append(random.choice(confidence_values))


train_df["score"] = new_confidence

# Following BERT requirements
train_df_bert = pd.DataFrame(
    {
        "id": range(len(train_df)),
        "label": train_df["score"],
        "alpha": ["a"] * train_df.shape[0],
        "text": train_df["sentence"].replace(r"\n", " ", regex=True),
    }
)

print(train_df_bert.head())
train_df_bert.to_csv(train_bert_csv_dir, sep="\t", index=False, header=False)
