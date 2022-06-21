"""Upload audios to Amazon s3.

This scripts uploads the extracted audio files on the Ubuntu server into
the Amazon s3 bucket.
"""
import os
import boto3


# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
excerpt_output_path = os.path.join(home_dir, "extracted_audio", "2")

# get an access token, local (from) directory, and S3 (to) directory
# from the command-line
local_directory = excerpt_output_path

bucket = 'extractedaudio'
destination = '2'
ACCESS_KEY = 'AKIA5JV4AUW3DNDSDB76'
SECRET_KEY = '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'

client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

# enumerate local files recursively
for root, dirs, files in os.walk(local_directory):

  for filename in files:

    # construct the full local path
    local_path = os.path.join(root, filename)

    # construct the full path
    relative_path = os.path.relpath(local_path, local_directory)
    s3_path = os.path.join(destination, relative_path)
    try:
        client.upload_file(local_path, bucket, s3_path)
        print('success!')
    except Exception as e:
        print(e)
