import boto3
import os

ACCESS_KEY = "*"
SECRET_KEY = "*"


s3 = boto3.resource(
    "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
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


home_dir = os.path.join("/home", "yyu")
local_dir = os.path.join(home_dir, "extracted_audios", "Samples_Benchmark_200")
download_s3_folder(
    bucket_name="extractedaudio", s3_folder="Samples_Benchmark_200", local_dir=local_dir
)
