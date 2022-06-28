
import os
import pandas as pd


home_dir = os.path.join("/home", "yyu")
file_dir = os.path.join(
    home_dir,
    "data_sheets",
    "confidence_dataframes",
)

folder_number = 0
csv_name = "confidence_dataframe_" + str(folder_number) + ".csv"
csv_path = os.path.join(file_dir, csv_name)

dataframe = pd.read_csv(csv_path)
questions_series = dataframe["sentence"]

print(type(questions_series))