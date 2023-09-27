# !pip install requests py7zr
import os
import requests
import py7zr

github_url = "https://github.com/Zabiullahkhan/Data_Science/raw/main/81_subcats_pb_ec_train_06092023.7z"

local_filename = "81_subcats_pb_ec_train_06092023.7z"
password = "1234"

response = requests.get(github_url)
with open(local_filename, "wb") as file:
    file.write(response.content)

with py7zr.SevenZipFile(local_filename, mode="r", password=password) as z:
    z.extractall()
os.remove(local_filename)
