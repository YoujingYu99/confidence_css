"""Generate the crowdsourcing_results_test_df, which is the same set as the human df. 
"""

import pandas as pd
import os


home_dir = os.path.join("/home", "youjing", "PersonalProjects", "confidence_css")
# Get datasheet path
folder_path = os.path.join(home_dir, "data", "label_results")
human_df = pd.read_csv(os.path.join(folder_path, "Human_Labels.csv"))
cleaned_results_df = pd.read_csv(
    os.path.join(folder_path, "Cleaned_Results_Renamed.csv")
)

cleaned_results_human = cleaned_results_df[
    cleaned_results_df.audio_url.isin(human_df.audio_url)
]
# save to csv
cleaned_results_human.to_csv(
    os.path.join(folder_path, "Cleaned_Results_Eval.csv"), index=False
)
