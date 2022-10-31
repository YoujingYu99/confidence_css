"""Upload audios to Amazon s3.
This scripts uploads the extracted audio files on the Ubuntu server into
the Amazon s3 bucket.
"""
import os
import boto3

home_dir = os.path.join("/home", "yyu")

# Only upload 0 to 6
folder_numbers = range(7)

for folder_number in folder_numbers:
    excerpt_output_path = os.path.join(
        home_dir, "extracted_audios", str(folder_number)
    )

    # get an access token, local (from) directory, and S3 (to) directory
    # from the command-line
    local_directory = excerpt_output_path

    # Set properties
    bucket = "extractedaudio"
    destination = str(folder_number)
    ACCESS_KEY = "*"
    SECRET_KEY = "*"

    client = boto3.client(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )

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
