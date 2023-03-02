# Docker
# python3 controller.py -iw "/mnt/VDev/Photos/000-InstantUpload/" -id "/mnt/Photos/005-PhotoBook/" -l "/mnt/VDev/Apps_Config/imageflow/" -s "/mnt/VDev/No_Share/secrets/imageflow/" -fc "/mnt/VDev/Apps_Config/imageflow/faceClassifier.pkl"

# python3 controller.py -iw "/mnt/Photos/000-InstantUpload/" -id "/mnt/Photos/005-PhotoBook/" -l "/mnt/Apps_Config/imageflow/" -s "/home/paolo/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"
# pip3 install requests face_recognition pillow scikit-learn scipy matplotlib clarifai_grpc pyexiv2
# pip3 install --upgrade setuptools protobuf

# NUC
# python3 controller.py -iw "/mnt/Photos/001-Process/IN/" -id "/mnt/Photos/001-Process/OUT/" -l "/mnt/Apps_Config/imageflow/" -s "/home/nuc/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl" -r true
# python3 controller.py -iw "/mnt/Documents/501-Knowledge/200- Personal/Food/Pizza alla Romana/" -id "/mnt/Photos/001-Process/OUT/" -l "/mnt/Apps_Config/imageflow/" -s "/home/nuc/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl" -r true --logLevel "INFO"


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

from config import fm_config


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
    os.system(
        f"exiftool -overwrite_original  -keywords='new keyword' {file_path}")
    print(f"Added keyword to {file_path}")
    pass


async def reverse_geo(file_path, args):
    print("reverse geo", file_path)
    pass


def extract_unique_keywords(lst):
    result = []
    for item in lst:
        if item == "None":
            continue
        sub_lst = eval(item)
        for keyword in sub_lst:
            if keyword not in result:
                result.append(keyword)
    return result


def add_to_list_if_not_exist(lst, items):
    items = items.replace("[", "").replace("]", "").replace("'", "")
    items = items.strip().split(", ")
    for item in items:
        # print("item", item)
        if item not in lst:
            lst.append(item)
    return lst


# function that processes files in the imagesQueue
async def process_media(imagesQueue, processMedia, logger, args):
    stop_timer = StopTimer.StopTimer()
    while True:
        file = await imagesQueue.get()
        file_path = str(Path(args.imagesWatchDirectory) / file)

        await asyncio.sleep(args.watchDelay)
        logger.info(f"|Workflow| Processing: {str(file_path)}")

        stop_timer.reset()
        stop_timer.start()

        # IMAGE WORKFLOW
        # TO DO: remove file tagging from each function adn add a separate one like in teh video wrokflow
        if file.lower().endswith(args.imageExtensions):
            if args.captionImage:
                await processMedia.caption_image(file_path, True)
            if args.tagImage:
                await processMedia.tag_image(file_path)
            if args.reverseGeotag:
                await processMedia.reverse_geotag(file_path)
            if args.classifyFaces:
                await processMedia.classify_faces(file_path, True)
            if args.ocrImage:
                await processMedia.ocr_image(file_path, True)
            if args.copyTagsToIPTC:
                await processMedia.copy_tags_to_IPTC(file_path)
            if args.idObjImage:
                await processMedia.id_obj_image(file_path, True)
            if args.moveFileVideo:
                await processMedia.move_file(file_path, args.imageDestinationDir)

        # VIDEO WORKFLOW
        if file.lower().endswith(args.videoExtensions):
            tmpFName = Path(file_path).stem
            tmpPath = fm_config.RAMDISK_DIR + str(tmpFName)
            logger.debug(
                f"|preProcessVideo| File: {file_path}. Temp path: {str(tmpPath)}")
            await processMedia.preProcessVideo(file_path)

            caption = []
            faces = []
            ocr = []
            objects = []

            for file in sorted(os.listdir(tmpPath), key=str.lower):
                file = os.path.abspath(os.path.join(Path(tmpPath), file))
                logger.info(f"|preProcessVideo| Processing: {file}")
                if file.lower().endswith(args.imageExtensions):
                    # TODO: add_to_list_if_not_exist replace with list(set(...))
                    if args.captionVideo:
                        try:
                            # Caption all the scene images
                            c = str(await processMedia.caption_image(file, False))
                            caption.append(c)
                        except Exception as e:
                            logger.error(
                                f"|captionVideo| Error: {e}")

                    if args.classifyFacesVideo:
                        try:
                            # Identify faces
                            f = str(await processMedia.classify_faces(file, False))
                            faces = add_to_list_if_not_exist(faces, f)
                        except Exception as e:
                            logger.error(
                                f"|classifyFacesVideo| Error: {e}")

                    if args.ocrVideo:
                        try:
                            # OCR texts in scene
                            o = str(await processMedia.ocr_image(file, False))
                            ocr.append(o)
                        except Exception as e:
                            logger.error(
                                f"|ocrVideo| Error: {e}")

                    if args.idObjVideo:
                        try:  # Identfy objects
                            ob = str(await processMedia.id_obj_image(file, False))
                            objects = add_to_list_if_not_exist(objects, ob)
                        except Exception as e:
                            logger.error(
                                f"|idObjVideo| Error: {e}")

                # Transcribe audio
                if file.lower().endswith(args.audioExtensions) and args.transcribeVideo:
                    try:
                        t = "\n |Transcribe|: "
                        t += str(await processMedia.transcribe(file))
                    except Exception as e:
                        logger.error(
                            f"|Transcribe| Error: {e}")

            processMedia.removeTempDirectory(file_path)

            logger.info(f"Caption: {caption}")
            logger.info(f"Faces: {faces}")
            logger.info(f"OCR: {ocr}")
            logger.info(f"Objects: {objects}")
            logger.info(f"Transcription: {t}|")

            kw = faces + objects
            d = ""
            o = "\n |OCR|: "

            for ele in caption:
                d += ". " + ele
            for ele in ocr:
                if ele != "None":
                    o += ". " + ele

            description = d + o + t
            try:
                await processMedia.write_keywords_metadata_to_video_file(
                    video_file_path=file_path, keywords=kw, description=description
                )
            except Exception as e:
                logger.error(
                    f"|write_keywords_metadata_to_video_file| Error: {e}")
            if args.moveFileVideo:
                await processMedia.move_file(file_path, args.videoDestinationFolder)

        imagesQueue.task_done()
        stop_timer.stop()
        logger.info(
            "Media processed in: " +
            str(stop_timer.duration()) + " " + str(file_path)
        )


async def recursive_listdir(path, recursive=True):
    if not recursive:
        return os.listdir(path)
    else:
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                result.append(os.path.join(root, name))
        return result

# function that watches a folder for new files


async def watch_folder(imagesQueue, args):
    processed_files = set()

    while True:
        # for file in os.listdir(args.imagesWatchDirectory):
        for file in await recursive_listdir(
            args.imagesWatchDirectory, recursive=args.watchRecursively
        ):
            if (
                file.lower().endswith(args.imageExtensions)
                or file.lower().endswith(args.videoExtensions)
            ) and file not in processed_files:
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
    # processMedia.create_ramdisk(fm_config.RAMDISK_DIR, fm_config.RAMDISK_SIZE_MB)

    # start the task that watches the folder for new files
    task1 = asyncio.create_task(watch_folder(imagesQueue, args))
    # start multiple tasks that process files in the imagesQueue
    task2 = [
        asyncio.create_task(process_media(
            imagesQueue, processMedia, logger, args))
        for _ in range(5)
    ]

    await asyncio.gather(task1, *task2)


if __name__ == "__main__":
    asyncio.run(main())
