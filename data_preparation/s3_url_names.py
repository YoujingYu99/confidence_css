"""Extract sw3 urls to csv.

This scripts extracts the urls of the audio files in sw3 and save to csv.
"""


import boto3
import numpy as np
import pandas as pd
import os
import random

home_dir = os.path.join("/home", "yyu")
folder_numbers = [0]
# folder_numbers = range(8)

num_tasks_per_HIT = 10
audio_column_name_list = []
for i in range(1, num_tasks_per_HIT+1, 1):
    audio_column_name = "audio_url_" + str(i)
    audio_column_name_list.append(audio_column_name)

for folder_number in folder_numbers:
    prefix = str(folder_number) + "/"

    ACCESS_KEY = "AKIA5JV4AUW3DNDSDB76"
    SECRET_KEY = "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"

    s3_client = boto3.client(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )

    bucket_name = "extractedaudio"
    # Empty list for url
    res = []
    for obj in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]:
        url = f'https://{bucket_name}.s3.eu-west-2.amazonaws.com/{obj["Key"]}'
        res.append(url)
        print(url)

    # Remove first element from list
    res.pop(0)
    # Remove duplicates from list and shuffle it
    res = list(set(res))
    random.shuffle(res)
    # Save to csv
    home_dir = os.path.join("/home", "yyu")
    datasheet_name = "input_" + str(folder_number) + ".csv"
    csv_output_path = os.path.join(home_dir, "data_sheets", "sw3_urls", datasheet_name)
    array_of_list = np.array_split(res, num_tasks_per_HIT)

    df = pd.DataFrame(array_of_list, columns=audio_column_name_list)
    df.to_csv(csv_output_path, index=False)
