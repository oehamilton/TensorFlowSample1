import os
import sys
import zipfile
import requests
import tempfile

# Get the system's temporary directory
temp_dir = tempfile.gettempdir()

# Define output paths for the ZIP files
output1 = os.path.join(temp_dir, "rps.zip")
output2 = os.path.join(temp_dir, "rps-test-set.zip")
print(f"Temporary directory: {temp_dir}")
print(f"Output file 1: {output1}")
print(f"Output file 2: {output2}")      


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

print(f"Files downloaded to {temp_dir}")


# Extract ZIP files
extract_dir = os.path.join(temp_dir, "rps_data")
print(f"Extracting files to {extract_dir}")


os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(output1, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
with zipfile.ZipFile(output2, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Files extracted to {extract_dir}")
# Define directories for rock, paper, and scissors
rock_dir = os.path.join(extract_dir,'rps\\rock')
paper_dir = os.path.join(extract_dir,'rps\\paper')
scissors_dir = os.path.join(extract_dir,'rps\\scissors')
print(f"Rock directory: {rock_dir}")
print(f"Paper directory: {paper_dir}")      
print(f"Scissors directory: {scissors_dir}")

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

