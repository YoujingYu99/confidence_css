"""Split total results to train, val and test datasets."""
from model_utils import split_to_train_val_test
import os

save_to_single_csv = False

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

split_to_train_val_test(home_dir)
