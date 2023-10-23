import requests
import os
import json
from pathlib import Path
import subprocess

# http://192.168.1.121:9999/mnt/Photos/005-PhotoBook/2023/09/
# http://192.168.1.121:9999/mnt/Photos/005-PhotoBook/2023/09/_DSC4552.jpg
# curl http://192.168.1.121:9997/predictions -X POST -H "Content-Type: application/json" -d '{"input": { "input_image": "http://192.168.1.121:9999/mnt/Photos/005-PhotoBook/2023/09/_DSC4552.jpg"}}'


def exif_rating(image_path):
    r = 0
    try:
        # Call exiftool targeting the 'Rating' tag and capture the output
        result = subprocess.run(
            ["exiftool", "-Rating", image_path], stdout=subprocess.PIPE, text=True
        )

        # Parse and print the rating value
        if "Rating" in result.stdout:
            r = result.stdout.split(":")[1].strip()
        else:
            r = 0

    except Exception as e:
        print(f"Error: {e}")
    return int(r)


def set_rating(rating, fname):
    # Construct the ExifTool command
    command = f"exiftool -overwrite_original -Rating={rating} '{fname}'"

    # Run the command
    subprocess.run(command, shell=True, check=True)

    print(f"Rating set to {rating} for {fname}")


IMAGES_SERVER_URL = "http://192.168.1.121:9999/mnt/Photos/005-PhotoBook/2023/09/"
OBJ_ID_API_URL = "http://192.168.1.121:9997/predictions"
OBJ_ID_HEADER = "content-type: application/json"

fname = "_DSC4552.jpg"
url = OBJ_ID_API_URL
payload = '{"input":{"image":"' + str(IMAGES_SERVER_URL) + str(fname) + '"}}'

import requests
import json


# Print the curl command for debugging
# print(f"curl {url} -X POST -H {json.dumps(headers)} -d {json.dumps(payload)}")

# Define the URL
url = "http://192.168.1.121:9997/predictions"

# Define the headers
headers = {"Content-Type": "application/json"}

# Assume IMAGES_SERVER_URL and fname are defined
IMAGES_SERVER_URL = "http://192.168.1.121:9999/"
fname = "mnt/Photos/005-PhotoBook/2023/09/_DSC4552.jpg"


def get_ranking(fname):
    output_value = 0
    # Define the payload
    payload = {"input": {"input_image": IMAGES_SERVER_URL + fname}}

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check the response
    if response.status_code == 200:
        data = response.json()

        # Check if the 'status' key is 'succeeded'
        if data.get("status") == "succeeded":
            output_value = data.get("output")
            # print("Success: The output value is", output_value)
        else:
            print("Failed: The task did not succeed. Status:", data.get("status"))
    else:
        print("Failed: HTTP Response Code", response.status_code, response.text)
    return output_value


def map_ranking(original_rank):
    # Inverting the original rank as 1 is great and 100 is bad
    inverted_rank = 101 - original_rank

    # Mapping the inverted rank to the range [1, 5]
    # Since inverted_rank is in [1, 100], we divide it by 20 to get it in the range [1, 5]
    mapped_rank = inverted_rank / 20

    # Rounding to the nearest integer to get a rank between 1 and 5
    final_rank = round(mapped_rank)

    # Ensuring that the final rank is within the bounds [1, 5]
    final_rank = max(1, min(final_rank, 5))

    return final_rank


def walk_path(root_path):
    """
    Walks through the given root_path and, for every image found, sends a request to the server and prints the ranking received in response.
    """
    # Use pathlib to handle paths
    root_path = Path(root_path)

    # Walk through the given path
    for current_path, _, files in os.walk(root_path):
        for file in files:
            # Check if the file is an image (by checking the file extension)
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Construct the full path of the image file
                image_path = Path(current_path) / file

                # Call the get_ranking function for the image
                hu_r = exif_rating(str(image_path))
                if hu_r == 0:
                    ai_r = get_ranking(str(image_path))
                    mapped = int(map_ranking(ai_r))
                    print(f"{image_path}, {ai_r}, {hu_r}, {mapped}, {(hu_r-mapped)}")
                    try:
                        set_rating(mapped, image_path)
                    except Exception as e:
                        print(e)


walk_path("/mnt/Photos/005-PhotoBook/2023/10")
