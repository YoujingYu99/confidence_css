from sklearn.dummy import DummyClassifier
from model_utils import load_audio_and_score_from_crowdsourcing_results, categorise_score
import os
import numpy as np

save_to_single_csv = False

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")

# Path for crowdsourcing results
crowdsourcing_results_df_path = os.path.join(
    home_dir, "data_sheets", "crowdsourcing_results", "test_crowd.csv",
)


print("start of application!")

# Read in individual csvs and load into a final dataframe
audio_df = load_audio_and_score_from_crowdsourcing_results(
    home_dir, crowdsourcing_results_df_path, save_to_single_csv
)

# Split to train, eval and test datasets.
df_train, df_val, df_test = np.split(
    audio_df.sample(frac=1, random_state=42),
    [int(0.8 * len(audio_df)), int(0.9 * len(audio_df))],
)

x = df_train["audio_array"].tolist()
y = df_train["score"].tolist()
y_cat = [categorise_score(i) for i in y]

x_test = df_test["audio_array"].tolist()
y_test = df_test["score"].tolist()
y_test_cat = [categorise_score(i) for i in y]

# Fit the dummy classifier on the data
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x, y_cat)

# Test the performance of the dummy classifier
dummy_clf.predict(x_test)
print("dummy test score", dummy_clf.score(x_test, y_test_cat))

