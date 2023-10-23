# Docker
# screen python3 /mnt/Software/200-Apps/imageflow/controller.py -iw "/mnt/Photos/000-InstantUpload/" -id "/mnt/Photos/005-PhotoBook/" -l "/mnt/Apps_Config/imageflow/" -s "/mnt/No_Share/secrets/imageflow/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

# Test
# python3 /mnt/Software/200-Apps/imageflow/controller.py -iw "/mnt/Photos/001-Process/IN/" -id "/mnt/Photos/001-Process/OUT/" -l "/mnt/Apps_Config/imageflow/" -s "/mnt/No_Share/secrets/imageflow/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

# DEV
# screen python3 /mnt/Software/200-Apps/imageflow/controller.py -iw "/mnt/Photos/000-InstantUpload/" -id "/mnt/Photos/005-PhotoBook/" -l "/mnt/Apps_Config/imageflow/" -s "/mnt/No_Share/secrets/imageflow/" -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

# ERRORS: video process colors are not captured correctley, GS reverse GEO does not work

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


def extract_unique_keywords(lst):
    result = set()
    for item in lst:
        if item == "None":
            continue
        sub_lst = eval(item)
        for keyword in sub_lst:
            result.add(keyword)
    return list(result)


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

        caption = ""
        comment = ""
        faces = []
        ocr = ""
        objects = []
        colors = []
        reverseGeo = []
        imageTags = []
        KW = []
        transcription = ""
        rank=""

        # IMAGE WORKFLOW
        # TO DO: remove file tagging from each function adn add a separate one like in the video workflow
        if file.lower().endswith(args.imageExtensions):
            if args.moveFileImage == "True":
                file_path = await processMedia.move_file(
                    file_path, args.imageDestinationDir
                )
            if args.tagImage == "True":
                try:
                    imageTags = await processMedia.tag_image(file_path)
                except Exception as e:
                    logger.error(f"|tag_image| Error: {e}")
            if args.reverseGeotag == "True":
                try:
                    reverseGeo = await processMedia.reverse_geotag(file_path)
                except Exception as e:
                    logger.error(f"|reverse_geotag| Error: {e}")

            if args.commentImage == "True":
                try:

                    # Comment image
                    comment = str(await processMedia.comment_image(file_path))
                except Exception as e:
                    logger.error(f"|Comment| Error: {e}")
            if args.captionImage == "True":
                try:
                    # Caption image
                    caption = str(await processMedia.caption_image(file_path))
                except Exception as e:
                    logger.error(f"|caption_image| Error: {e}")
            if args.classifyFaces == "True":
                try:
                    # Identify faces
                    f = await processMedia.classify_faces(file_path)
                    if len(f):
                        faces = f
                except Exception as e:
                    logger.error(f"|classifyFacesImage| Error: {e}")

            if args.ocrImage == "True":
                try:
                    # OCR texts in scene
                    ocr = str(await processMedia.ocr_image(file_path, returnTag=False))
                except Exception as e:
                    logger.error(f"|ocrImage| Error: {e}")

            if args.idObjImage == "True":
                try:
                    # Identfy objects
                    objects = await processMedia.id_obj_image(
                        file_path, returnTag=False
                    )
                except Exception as e:
                    logger.error(f"|idObjImage| Error: {e}")

            if args.getColorsImage == "True":
                try:
                    colors = await processMedia.get_top_colors(file_path, n=5)

                except Exception as e:
                    logger.error(f"|getColorsImage| Error: {e}")

            if args.rateImage == "True":
                try:
                    rate = await processMedia.generate_rating(file_path)

                except Exception as e:
                    logger.error(f"|rateImage| Error: {e}")

            if args.writeTagToImage == "True":
                if reverseGeo is not None:
                    KW += reverseGeo
                if imageTags is not None:
                    KW += imageTags
                if objects is not None:
                    KW += objects
                if colors is not None:
                    KW += colors
                if faces is not None:
                    KW += faces
                noOCR = "None"
                if ocr == noOCR:
                    ocr = ""
                if comment:
                    caption = caption + "\n" + comment

                await processMedia.write_keywords_metadata_to_image_file(
                    file_path,
                    keywords=KW,
                    caption=str(caption) + "\n" + str(comment),
                    subject=ocr,
                )
                if fm_config.COPY_TAGS_TO_IPTC == True:
                    await processMedia.copy_tags_to_IPTC(file_path)

        # VIDEO WORKFLOW
        if file.lower().endswith(args.videoExtensions):
            videoWorkflowSuccess = True
            try:
                # We first move the video because changing keywords in videos creates a new video thus losing the info of the original creation date
                if args.moveFileVideo == "True":
                    file_path = await processMedia.move_file(
                        file_path, args.videoDestinationDir
                    )
                tmpFName = Path(file_path).stem
                tmpPath = fm_config.RAMDISK_DIR + str(tmpFName)
                logger.info(
                    f"|preProcessVideo| File: {file_path}. Temp path: {str(tmpPath)}"
                )
                await processMedia.preProcessVideo(file_path)

                caption = []
                faces = []
                ocr = []
                objects = []
                transcription = ""
                imageTags=""
                reverseGeo=""
                colors=""
                rate=""

                for file in sorted(os.listdir(tmpPath), key=str.lower):
                    file = os.path.abspath(os.path.join(Path(tmpPath), file))
                    logger.info(f"|preProcessVideo| Processing: {file}")
                    if file.lower().endswith(args.imageExtensions):
                        c = []
                        t = []
                        f = []
                        o = []
                        colors = []
                        ob = ""
                        # TODO: add_to_list_if_not_exist replace with list(set(...))
                        if args.captionVideo == "True":
                            try:
                                # Caption all the scene images
                                c = str(await processMedia.caption_image(file, False))
                                caption.append(c)
                            except Exception as e:
                                logger.error(f"|captionVideo| Error: {e}")

                        if args.classifyFacesVideo == "True":
                            try:
                                # Identify faces
                                f = str(await processMedia.classify_faces(file))
                                faces = add_to_list_if_not_exist(faces, f)
                            except Exception as e:
                                logger.error(f"|classifyFacesVideo| Error: {e}")

                        if args.ocrVideo == "True":
                            try:
                                # OCR texts in scene
                                o = str(
                                    await processMedia.ocr_image(file, returnTag=False)
                                )
                                ocr.append(o)
                            except Exception as e:
                                logger.error(f"|ocrVideo| Error: {e}")

                        if args.idObjVideo == "True":
                            try:  # Identfy objects
                                ob = str(
                                    await processMedia.id_obj_image(
                                        file, returnTag=False
                                    )
                                )
                                objects = add_to_list_if_not_exist(objects, ob)
                            except Exception as e:
                                logger.error(f"|idObjVideo| Error: {e}")
                        if args.getColorsVideo == "True":
                            colors = ""
                            try:
                                colors = await processMedia.get_top_colors(file, n=5)

                            except Exception as e:
                                logger.error(f"|getColorsVideo| Error: {e}")

                    # Transcribe audio
                    if (
                        file.lower().endswith(args.audioExtensions)
                        and args.transcribeVideo == "True"
                    ):
                        try:
                            transcription = ""
                            transcription += str(
                                await processMedia.transcribe(file, returnTag=False)
                            )
                        except Exception as e:
                            logger.error(f"|Transcribe| Error: {e}")
            except Exception as e:
                videoWorkflowSuccess = False
                logger.error(f"|VideoWorkflow| Error: {e}")

            processMedia.removeTempDirectory(file_path)
            if videoWorkflowSuccess:
                KW = faces + objects + colors
                d = ""
                o = ""

                for ele in caption:
                    d += ". " + ele
                for ele in ocr:
                    if ele != "None":
                        o += ". " + ele

                description = f"{fm_config.TRANSCRIBE_TAG_OPEN}{transcription}{fm_config.TRANSCRIBE_TAG_CLOSE}{fm_config.CAPTION_TAG_OPEN}{d}{fm_config.CAPTION_TAG_CLOSE}{fm_config.OBJECTS_TAG_OPEN}{objects}{fm_config.OBJECTS_TAG_CLOSE}{fm_config.OCR_TAG_OPEN}{o}{fm_config.OCR_TAG_CLOSE}{fm_config.FACES_TAG_OPEN}{faces}{fm_config.FACES_TAG_CLOSE}{fm_config.COLORS_TAG_OPEN}{colors}{fm_config.COLORS_TAG_CLOSE}"
                try:
                    await processMedia.write_keywords_metadata_to_video_file(
                        file_path=file_path, keywords=KW, description=description
                    )
                    # Creates a sidecar file for the video and changes permisisons accorgingly
                    sidecar_file = f"{file_path}.txt"
                    with open(sidecar_file, "w") as f:
                        f.write(description)
                    os.chmod(sidecar_file, 0o777)
                except Exception as e:
                    logger.error(f"|write_keywords_metadata_to_video_file| Error: {e}")

            else:
                logger.error(
                    f"|VideoWorkflow| =====> WORKFLOW ERROR <===== File '{file_path}' not changed"
                )

        logger.info(f"Transcription: {transcription}|")
        logger.info(f"Caption: {caption}")
        logger.info(f"Keywords: {imageTags}")
        logger.info(f"Reverse Geo: {reverseGeo}")
        logger.info(f"Faces: {faces}")
        logger.info(f"OCR: {ocr}")
        logger.info(f"Objects: {objects}")
        logger.info(f"Colors: {colors}|")
        logger.info(f"Rate: {rate}|")

        imagesQueue.task_done()
        stop_timer.stop()
        logger.info(
            "Media processed in: " + str(stop_timer.duration()) + " " + str(file_path)
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
        asyncio.create_task(process_media(imagesQueue, processMedia, logger, args))
        for _ in range(5)
    ]

    await asyncio.gather(task1, *task2)


if __name__ == "__main__":
    asyncio.run(main())
