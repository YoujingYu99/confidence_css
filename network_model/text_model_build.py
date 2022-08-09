"""Use pre-trained BERT model on the text for classification.
Extract the text and confidence score to a csv. Then from the csv train the
BERT model for confidence classification.
"""
from transformers import BertTokenizer
from model_utils import *
from models import *

# # Memory issues
# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)

text_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Load dataset
home_dir = os.path.join("/home", "yyu")
train_csv_dir = os.path.join(home_dir, "data_sheets", "train_new2.csv")
text_df = pd.read_csv(train_csv_dir, nrows=1000)
print(text_df.head())
df_train, df_val, df_test = np.split(
    text_df.sample(frac=1, random_state=42),
    [int(0.8 * len(text_df)), int(0.9 * len(text_df))],
)
print(len(df_train), len(df_val), len(df_test))

# Decide on Epoch and model
EPOCHS = 5
text_model = BertClassifier()
LR = 1e-6
batch_size = 3

# Train model
train_text(text_model, text_tokenizer, df_train, df_val, LR, EPOCHS, batch_size)
evaluate_text(text_model, df_test, text_tokenizer, batch_size)
