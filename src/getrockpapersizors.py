import os
import zipfile
import requests
import tempfile

# Get the system's temporary directory
temp_dir = tempfile.gettempdir()

# Define output paths for the ZIP files
output1 = os.path.join(temp_dir, "rps.zip")
output2 = os.path.join(temp_dir, "rps-test-set.zip")

# Download the files
url1 = "https://storage.googleapis.com/learning-datasets/rps.zip"
url2 = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"

response1 = requests.get(url1)
if response1.status_code == 200:
    with open(output1, "wb") as f:
        f.write(response1.content)
else:
    raise Exception(f"Download failed for {url1}: Status code {response1.status_code}")

response2 = requests.get(url2)
if response2.status_code == 200:
    with open(output2, "wb") as f:
        f.write(response2.content)
else:
    raise Exception(f"Download failed for {url2}: Status code {response2.status_code}")

# Extract ZIP files
extract_dir = os.path.join(temp_dir, "rps_data")
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(output1, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
with zipfile.ZipFile(output2, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Files extracted to {extract_dir}")