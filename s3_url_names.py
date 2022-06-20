"""Extract sw3 urls to csv.

This scripts extracts the urls of the audio files in sw3 and save to csv.
"""


import boto3
import pandas as pd
import os

ACCESS_KEY = 'AKIA5JV4AUW3DNDSDB76'
SECRET_KEY = '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'

s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY)

bucket_name = 'extractedaudio'
# Empty list for url
res = []
for obj in \
s3_client.list_objects_v2(Bucket=bucket_name, Prefix="0/")[
    'Contents']:
    url = f'https://{bucket_name}.s3.amazonaws.com/{obj["Key"]}'
    res.append(url)
    print(url)

# Remove first element from list
res.pop(0)
# Remove duplicates from list
res = list(set(res))
# Save to csv
home_dir = os.path.join("/home", "yyu")
csv_output_path = os.path.join(home_dir, "data_sheets", "input_0.csv")
df = pd.DataFrame(res, columns=['audio_url'])
df.to_csv(csv_output_path, index=False)