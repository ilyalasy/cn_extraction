import gzip
import shutil
import requests
from tqdm import tqdm
from filtering import filter

CN_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"

filename = CN_URL.split("/")[-1]
# Streaming, so we can iterate over the response.
response = requests.get(CN_URL, stream=True)
total_size_in_bytes= int(response.headers.get('content-length', 0))
block_size = 1024 #1 Kibibyte
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
print('Downloading ConceptNet')
with open(filename, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()
if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    raise Exception("ERROR, something went wrng")
print('Downloading finished!')
print('Unpacking...')
with gzip.open(filename, 'rb') as f_in:
    with open(filename.replace('.gz',''), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print('Unpacked!')
filter()