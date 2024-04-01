"""Handlers for upload, download and names with Amazon s3.
This scripts uploads the extracted audio files on the Ubuntu server into
the Amazon s3 bucket.
"""

import os
import boto3
import random
import math
import numpy as np
import pandas as pd

ACCESS_KEY = "*"
SECRET_KEY = "*"

client = boto3.client(
    "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
)

home_dir = os.path.join("/home", "yyu")

# Upload to s3; Only upload 0 to 6
folder_numbers = range(7)

for folder_number in folder_numbers:
    excerpt_output_path = os.path.join(home_dir, "extracted_audios", str(folder_number))

    # get an access token, local (from) directory, and S3 (to) directory
    # from the command-line
    local_directory = excerpt_output_path

    # Set properties
    bucket = "extractedaudio"
    destination = str(folder_number)

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):

        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)
            # construct the full remote path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            try:
                client.upload_file(local_path, bucket, s3_path)
                print("success!")
            except Exception as e:
                print(e)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = client.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


## Download s3
local_dir = os.path.join(home_dir, "extracted_audios", "Benchmark_Samples")
download_s3_folder(
    bucket_name="extractedaudio", s3_folder="Benchmark_Samples", local_dir=local_dir
)

## Extracts the urls of the audio files in sw3 and save to csv. Since we set 10 audios per HIT plus 2 test questions, we have 12 audios per HIT.
folder_numbers = [0]
# folder_numbers = range(7)

num_tasks_per_HIT = 10
audio_column_name_list = []

# Set the names of columns
for i in range(1, num_tasks_per_HIT + 1, 1):
    audio_column_name = "audio_url_" + str(i)
    audio_column_name_list.append(audio_column_name)

for folder_number in folder_numbers:
    print(folder_number)
    prefix = str(folder_number) + "/"

    bucket_name = "extractedaudio"
    # Empty list for url
    res = []
    for obj in client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]:
        url = f'https://{bucket_name}.s3.eu-west-2.amazonaws.com/{obj["Key"]}'
        res.append(url)
        print(url)

    # Remove first element from list
    res.pop(0)
    # Remove duplicates from list and shuffle it
    res = list(set(res))
    random.shuffle(res)
    # Get number of audio urls in res
    total_number_urls = len(res)
    # Get number of HITs we can get
    number_HIT = math.floor(total_number_urls / num_tasks_per_HIT)
    # Only choose some audios so we can segment into 10 per HIT / Leave the last incomplete out
    res = res[: number_HIT * num_tasks_per_HIT]
    # Save to csv
    datasheet_name = "input_" + str(folder_number) + ".csv"
    csv_output_path = os.path.join(home_dir, "data_sheets", "sw3_urls", datasheet_name)
    array_of_list = np.array_split(res, number_HIT)

    # A list of 100 lists, each has 10 values
    print(len(array_of_list[0]))
    df = pd.DataFrame(array_of_list, columns=audio_column_name_list)
    df.to_csv(csv_output_path, index=False)
