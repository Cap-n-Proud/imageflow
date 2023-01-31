# python3 controller.py -iw "/mnt/Photos/001-InstantUpload/" -id "/mnt/Photos/005-PhotoBook/" -l "/mnt/Apps_Config/imageflow/" -s "/home/paolo/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

# pip3 install requests face_recognition pillow scikit-learn scipy matplotlib clarifai_grpc pyexiv2
# pip install protobuf==3.20. pip3 install numpy==1.23.0
# pip3 install --upgrade setuptools

import asyncio
import os
import datetime
from asyncio import Queue
from pathlib import Path
from datetime import datetime
import json

import logging
import sys
import argparse

# Importing the script to parse all the arguemnts
import parser

import ProcessMedia
import StopTimer


def init(args):

    logger = logging.getLogger()
    logger.setLevel(eval("logging." + args.logLevel))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(eval("logging." + args.logLevel))

    FORMAT = "[%(asctime)s][%(name)s][%(levelname)s][%(message)s]"
    formatter = logging.Formatter(FORMAT)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(args.logFilePath + args.logName)
    file_handler.setLevel(eval("logging." + args.logLevel))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


# function that adds keyword and caption to file's metadata
async def add_metadata(file_path, args):
    print(file_path)
    # code to add keyword and caption to file's metadata
    os.system(f"exiftool -overwrite_original  -keywords='new keyword' {file_path}")
    print(f"Added keyword to {file_path}")
    pass


async def reverse_geo(file_path, args):
    print("reverse geo", file_path)
    pass


# function that processes files in the imagesQueue
async def process_images(imagesQueue, processMedia, logger, args):
    stop_timer = StopTimer.StopTimer()
    while True:
        file = await imagesQueue.get()
        file_path = str(Path(args.imagesWatchDirectory) / file)

        await asyncio.sleep(args.watchDelay)
        logger.info("Processing: " + str(file_path))
        stop_timer.reset()
        stop_timer.start()
        if args.captionImage:
            await processMedia.caption_image(file_path)
        if args.tagImage:
            await processMedia.tag_image(file_path)
        if args.reverseGeotag:
            await processMedia.reverse_geotag(file_path)
        if args.classifyFaces:
            await processMedia.classify_faces(file_path)
        if args.ocrImage:
            await processMedia.ocr_image(file_path)
        if args.copyTagsToIPTC:
            await processMedia.copy_tags_to_IPTC(file_path)
        if args.moveFile:
            await processMedia.move_file(file_path, args.imageDestinationDir)

        # await processMedia.info()
        imagesQueue.task_done()
        stop_timer.stop()
        logger.info(
            "Media processed in: " + str(stop_timer.duration()) + " " + str(file_path)
        )


# function that watches a folder for new files
async def watch_folder(imagesQueue, args):
    processed_files = set()

    while True:
        for file in os.listdir(args.imagesWatchDirectory):
            if (
                file.lower().endswith(args.imageExtensions)
                and file not in processed_files
            ):
                imagesQueue.put_nowait(file)
                processed_files.add(file)
                # logger.debug("Added to queue:" + str(file))

        await asyncio.sleep(1)


async def main():
    imagesQueue = Queue()

    args, remaining_args = parser.parser.parse_known_args()
    # sys.path.append(args.configFileDirectory)
    # from config import fm_config

    FORMAT = "[%(asctime)s][%(name)s][%(levelname)s][%(message)s]"

    logger = logging.getLogger()
    logger.setLevel(eval("logging." + args.logLevel))
    formatter = logging.Formatter(FORMAT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(eval("logging." + args.logLevel))
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(args.logFilePath + args.logName)
    file_handler.setLevel(eval("logging." + args.logLevel))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    ProcessMedia.logger = logger
    processMedia = ProcessMedia.ProcessMedia(args)
    logger.info("Server successfully started:" + str(datetime.now()))
    logger.info("Settings:" + str(vars(args)))

    # start the task that watches the folder for new files
    task1 = asyncio.create_task(watch_folder(imagesQueue, args))
    # start multiple tasks that process files in the imagesQueue
    task2 = [
        asyncio.create_task(process_images(imagesQueue, processMedia, logger, args))
        for _ in range(5)
    ]

    await asyncio.gather(task1, *task2)


if __name__ == "__main__":
    asyncio.run(main())
