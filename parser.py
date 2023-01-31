import argparse
import sys
from config import fm_config

parser = argparse.ArgumentParser(description="Example script")

# Group arguments by their functionality
folders_group = parser.add_argument_group("folder options")
extensions_group = parser.add_argument_group("extensions options")
workflow_group = parser.add_argument_group("image workflow options")
settings_group = parser.add_argument_group("settings options")

workflow_group.add_argument(
    "-wc",
    "--captionImage",
    help=f"Caption image [True/False]. Default is: {fm_config.CAPTION_IMAGE}",
    default=fm_config.CAPTION_IMAGE,
)
workflow_group.add_argument(
    "-wm",
    "--moveFile",
    help=f"Move file [True/False]. Default is: {fm_config.MOVE_FILE}",
    default=fm_config.MOVE_FILE,
)
workflow_group.add_argument(
    "-wt",
    "--tagImage",
    help=f"Tag image [True/False]. Default is: {fm_config.TAG_IMAGE}",
    default=fm_config.TAG_IMAGE,
)
workflow_group.add_argument(
    "-wr",
    "--reverseGeotag",
    help=f"Reverse geocoding image [True/False]. Default is: {fm_config.REVERSE_GEOTAG}",
    default=fm_config.REVERSE_GEOTAG,
)
workflow_group.add_argument(
    "-wf",
    "--classifyFaces",
    help=f"Find and classify faces in image [True/False]. Default is: {fm_config.CLASSIFY_FACES}",
    default=fm_config.CLASSIFY_FACES,
)
workflow_group.add_argument(
    "-wo",
    "--ocrImage",
    help=f"Perform scene OCR in image [True/False]. Default is: {fm_config.OCR_IMAGE}",
    default=fm_config.OCR_IMAGE,
)
workflow_group.add_argument(
    "-wi",
    "--copyTagsToIPTC",
    help=f"Copy Exif tags to IPTC space. Default is: {fm_config.COPY_TAGS_TO_IPTC}",
    default=fm_config.COPY_TAGS_TO_IPTC,
)


folders_group.add_argument(
    "-c",
    "--configFileDirectory",
    help=f"Directory containing log file. Default is: {fm_config.CONFIG_PATH}",
    default=fm_config.CONFIG_PATH,
)
folders_group.add_argument(
    "-iw",
    "--imagesWatchDirectory",
    help=f"Directory containing images to monitor. Default is: {fm_config.IMAGES_WATCH_DIRECTORY}",
    default=fm_config.IMAGES_WATCH_DIRECTORY,
)
folders_group.add_argument(
    "-dw",
    "--documentsWatchDirectory",
    help=f"Directory containing documents to monitor. Default is: {fm_config.IMAGES_WATCH_DIRECTORY}",
    default=fm_config.IMAGES_WATCH_DIRECTORY,
)
folders_group.add_argument(
    "-sc",
    "--screenshotsDestFolder",
    help=f"Path to screenshot. Default is: {fm_config.SCREENSHOT_DEST_FOLDER}",
    default=fm_config.SCREENSHOT_DEST_FOLDER,
)

folders_group.add_argument(
    "-id",
    "--imageDestinationDir",
    help=f"Path to image destination. Default is: {fm_config.IMAGE_DEST_DIRECTORY}",
    default=fm_config.IMAGE_DEST_DIRECTORY,
)

folders_group.add_argument(
    "-l",
    "--logFilePath",
    help=f"Path to log file. Default is: {fm_config.LOG_FILEPATH}",
    default=fm_config.LOG_FILEPATH,
)

folders_group.add_argument(
    "-s",
    "--secretsPath",
    help=f"Path to secrets file. Default is: {fm_config.SECRETS_PATH}",
    default=fm_config.SECRETS_PATH,
)

folders_group.add_argument(
    "-vd",
    "--videoDestinationFolder",
    help=f"Path to video destination folder. Default is: {fm_config.VIDEO_DEST_FOLDER}",
    default=fm_config.VIDEO_DEST_FOLDER,
)
folders_group.add_argument(
    "-fc",
    "--faceClassifierFile",
    help=f"Face classifier file. Default is: {fm_config.FACE_CLASSIFIER_FILE}",
    default=fm_config.FACE_CLASSIFIER_FILE,
)

extensions_group.add_argument(
    "-ie",
    "--imageExtensions",
    help=f"Extensions of files to watch ({fm_config.IMAGE_EXTENSIONS})",
    default=fm_config.IMAGE_EXTENSIONS,
)
extensions_group.add_argument(
    "-ve",
    "--videoExtensions",
    help=f"Extensions of files to watch ({fm_config.VIDEO_EXTENSIONS})",
    default=fm_config.VIDEO_EXTENSIONS,
)
extensions_group.add_argument(
    "-de",
    "--documentsExtensions",
    help=f"Extensions of files to watch ({fm_config.DOC_EXTENSIONS})",
    default=fm_config.DOC_EXTENSIONS,
)
extensions_group.add_argument(
    "-m",
    "--moveExt",
    help=f"Extensions of files to move. ({fm_config.MOVE_EXT})",
    default=fm_config.MOVE_EXT,
)

settings_group.add_argument(
    "-d",
    "--watchDelay",
    help=f"Watch delay ({fm_config.WATCH_DELAY})",
    default=fm_config.WATCH_DELAY,
    type=int,
)

settings_group.add_argument(
    "--logLevel",
    help=f"Log level. Default is: {fm_config.LOG_LEVEL}",
    default=fm_config.LOG_LEVEL,
)

settings_group.add_argument(
    "--logName",
    help=f"Log name. Default is: {fm_config.LOG_NAME}",
    default=fm_config.LOG_NAME,
)


# parse the arguments
args = parser.parse_args()
sys.path.append(args.configFileDirectory)
from config import fm_config