import asyncio
import os
from datetime import datetime

#
# async def add_tag_to_metadata(file_path):
#     creation_time = os.path.getctime(file_path)
#     current_time = datetime.now().timestamp()
#     time_diff = current_time - creation_time
#
#     # Wait for 10 seconds after the file's creation before adding the tag
#     if time_diff < 10:
#         await asyncio.sleep(10 - time_diff)
#
#     # Add the tag to the file's metadata here
#     # For example: os.system(f'exiftool -keywords+="tag" {file_path}')
#
#
# async def process_folder(folder_path):
#     while True:
#         for file_name in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, file_name)
#             if os.path.isfile(file_path):
#                 asyncio.create_task(add_tag_to_metadata(file_path))
#         await asyncio.sleep(10)
#
#
# folder_path = "/path/to/folder"
# asyncio.run(process_folder(folder_path))
#
import asyncio
import os
from PIL import Image


async def add_keyword(file_path):
    await asyncio.sleep(10)  # wait 10 seconds before processing the file
    with Image.open(file_path) as img:
        img.save(file_path, "jpeg", keywords=["keyword"])
        print("KW added")


async def watch_folder(folder_path):
    while True:
        await asyncio.sleep(1)  # check for new files every second
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                asyncio.create_task(add_keyword(file_path))
                print("task added")


if __name__ == "__main__":
    folder_path = "/home/paolo/Downloads/"  # replace with the actual path to the folder
    asyncio.run(watch_folder(folder_path))
