"""Set a number of samples to be extracted from each folder and move them to
the chosen folder.
"""
import os
import random
import shutil
import boto3
import pandas as pd

folder_numbers = [7]
# folder_numbers = range(8)
no_of_files = 200
#
home_dir = os.path.join("/home", "yyu")

# Get datasheet path
old_path = os.path.join(home_dir, "extracted_audio")
new_path = os.path.join(home_dir, "extracted_audio_samples_benchmark_200")
existed_path = os.path.join(home_dir, "extracted_audio_samples_benchmark")

for i in folder_numbers:
    source = os.path.join(old_path, str(i))
    dest = new_path
    files = os.listdir(source)

    for file_name in random.sample(files, no_of_files):
        # Only move if file does not exist in directory
        if not os.path.isfile(os.path.join(existed_path, file_name) + ".mp3"):
            shutil.move(os.path.join(source, file_name), dest)
    print("Done 1 folder!")

# # Upload to amazon s3
# excerpt_output_path = os.path.join(
#     home_dir, "extracted_audio_samples_benchmark"
# )
#
# # get an access token, local (from) directory, and S3 (to) directory
# # from the command-line
# local_directory = excerpt_output_path
#
# # Set properties
# bucket = "extractedaudio"
# destination = "samples_benchmark"
# ACCESS_KEY = "AKIA5JV4AUW3DNDSDB76"
# SECRET_KEY = "qDjIKdmO7MGcAG3lB32AQt36Udo45kC1GtoYhPZ+"
#
# client = boto3.client(
#     "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
# )

# # enumerate local files recursively
# for root, dirs, files in os.walk(local_directory):
#
#     for filename in files:
#         # construct the full local path
#         local_path = os.path.join(root, filename)
#         # construct the full remote path
#         relative_path = os.path.relpath(local_path, local_directory)
#         s3_path = os.path.join(destination, relative_path)
#         try:
#             client.upload_file(local_path, bucket, s3_path)
#             print("success!")
#         except Exception as e:
#             print(e)


# prefix = "samples_benchmark_200" + "/"
#
# ACCESS_KEY = "AKIA5JV4AUW3DNDSDB76"
# SECRET_KEY = "qDjIKdmO7MGcAG3lB32AQt36Udo45kC1GtoYhPZ+"
#
# s3_client = boto3.client(
#     "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
# )
#
# bucket_name = "extractedaudio"
# # Empty list for url
# res = []
# for obj in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]:
#     url = f'https://{bucket_name}.s3.eu-west-2.amazonaws.com/{obj["Key"]}'
#     res.append(url)
#     print(url)
#
# # Remove first element from list
# res.pop(0)
# # Remove duplicates from list and shuffle it
# res = list(set(res))
# random.shuffle(res)
# # Save to csv
# home_dir = os.path.join("/home", "yyu")
# datasheet_name = "samples_benchmark_200.csv"
# csv_output_path = os.path.join(home_dir, "data_sheets", "sw3_urls", datasheet_name)
#
# df = pd.DataFrame(res)
# df.to_csv(csv_output_path, index=False)
