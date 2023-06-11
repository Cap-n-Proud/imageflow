# Application configuration File
################################

# Directory To Watch, If not specified, the following value will be considered explicitly.
# DOCUMENTS_WATCH_DIRECTORY = "/home/paolo/Downloads/"
IMAGES_WATCH_DIRECTORY = "/mnt/Photos/000-InstantUpload/"

# Delay Between Watch Cycles In Seconds
WATCH_DELAY = 3

# TO BE IMPLEMENTED: TO BE ABLE TO RESUME FILES
# MIN_AGE = 300

# Check The WATCH_DIRECTORY and its children
WATCH_RECURSIVELY = True

# whether to watch for directory events
# DO_WATCH_DIRECTORIES = True

# Patterns of the files to watch
IMAGE_EXTENSIONS = (".jpg", "jpeg", ".webp", ".heic",
                    ".png", ".bmp", ".gif", ".tif", ".tiff", ".arw", ".dng")
# IMAGE_EXTENSIONS = IMAGE_EXTENSIONS + "," + IMAGE_EXTENSIONS.upper()

VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi")
AUDIO_EXTENSIONS = (".aac", ".mp3")
DOC_EXTENSIONS = (
    ".txt",
    ".trc",
    ".log",
    ".pdf",
    ".py",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".csv",
    ".dat",
    ".bat",
    ".sh",
    ".jar",
    ".htm",
    ".html",
    ".css",
    ".js",
)


# EXCEPTION_PATTERN = ["EXCEPTION", "FATAL", "ERROR"]

LOG_NAME = "imageflow_dev.log"
LOG_FILEPATH = "/mnt/Apps_Config/imageflow/"
LOG_LEVEL = "INFO"
SCREENSHOT_DEST_FOLDER = "/mnt/Photos/001-Screenshots/"
IMAGE_DEST_DIRECTORY = "/mnt/Photos/005-PhotoBook/"
VIDEO_DEST_DIRECTORY = "/mnt/Photos/010-Videos/"

SECRETS_PATH = "/home/nuc/"
CONFIG_PATH = "/mnt/Software/200-Apps/imageflow/config/"
#SECRETS_PATH = "/app/imageflow/imageflow_secrets/"
#CONFIG_PATH = "/app/imageflow/config/"


NO_PERSON_KW = "no_person"
SCREENSHOT_TAG = "Comment"
OCR_TIMEOUT = 60

MOVE_EXT_IMG = ".jpg,.png,.tiff,.arw"
MOVE_EXT_VIDEO = ".mov,.mp4,.mkv,"
MOVE_EXT = MOVE_EXT_IMG + MOVE_EXT_VIDEO + \
    MOVE_EXT_IMG.upper() + MOVE_EXT_VIDEO.upper()
PRE_PROCESS_ORIG = "/mnt/Photos/000-InstantUpload/"
# PRE_PROCESS_DEST = "/mnt/Photos/000-Process/"

# MEDIA_QUEUE = "/mnt/imageflow/media_queue.csv"

# Need to be long enough to allow the completion of the file transfer
# MEDIA_PROCESS_AGE = 60

# MEDIA_PROCESS_CHECK = 10


# --------------------------- Workflow config ---------------------------

CAPTION_IMAGE = True
TAG_IMAGE = True
REVERSE_GEOTAG = True
CLASSIFY_FACES = True
OCR_IMAGE = True
COPY_TAGS_TO_IPTC = False
GET_COLORS_IMAGE = True
ID_OBJ = True
MOVE_FILE_IMAGE = False
WRITE_TAGS_TO_IMAGE = True

# CAPTION_IMAGE = False
# TAG_IMAGE = False
# REVERSE_GEOTAG = False
# CLASSIFY_FACES = False
# OCR_IMAGE = True
# COPY_TAGS_TO_IPTC = False
# GET_COLORS_IMAGE = False
# ID_OBJ = False
# MOVE_FILE_IMAGE = False
# WRITE_TAGS_TO_IMAGE = True

CAPTION_VIDEO = True
REVERSE_GEOTAG_VIDEO = True
CLASSIFY_FACES_VIDEO = True
OCR_VIDEO = True
ID_OBJ_VIDEO = True
MOVE_FILE_VIDEO = False
GET_COLORS_VIDEO = True
TRANSCRIBE_VIDEO = True

# CAPTION_VIDEO = False
# REVERSE_GEOTAG_VIDEO = False
# CLASSIFY_FACES_VIDEO = False
# OCR_VIDEO = False
# ID_OBJ_VIDEO = False
# MOVE_FILE_VIDEO = False
# GET_COLORS_VIDEO = True
# TRANSCRIBE_VIDEO = False
# --------------------------- Image processor config ---------------------------
FACE_TRAINING_DIR = "/mnt/Photos/990-Faces/known_faces/"
REVERSE_GEO_URL = "https://api.opencagedata.com/geocode/v1/json?q="
CROP_FACE_SOURCE_DIR = "/mnt/Photos/005-PhotoBook/2022/"
#CROP_FACE_DEST_DIR = "/mnt/Photos/000-Process/faces/"
FACE_CLASSIFIER_FILE = "/mnt/Apps_Config/imageflow/faceClassifier.pkl"
# FACE_CLASSIFIER_DIR = "/mnt/Software/200-Apps/ZZZ/filemaster/"
FACE_CLASSIFIER_TRAIN_DIR = "/mnt/Photos/990-Faces/known_faces/"
UNKNOWN_FACE_FOLDER = "/mnt/Photos/990-Faces/unknown_faces/"
UNKNOWN_FACE_NAME = "unknown_face"
FACE_DISTANCE = 0.5


JSON_FOLDER = "/mnt/Apps_Config/imageflow/json/"

CAPTION_API_URL = "http://192.168.1.163:5000/predictions"
CAPTION_API_URL = "http://192.168.1.121:9111/predictions"
# NOTE: mount on the server must be the same as the photo foler e.g.: IMAGES_SERVER_URL + "/mnt/Photos"
IMAGES_SERVER_URL = "http://192.168.1.121:9999"
IMAGE_CAPTION_HEADER = {"content-type": "application/json"}

# --------------------------- Image OCR --------------------------
OCR_CAPTION_HEADER = {"content-type": "application/json"}
OCR_API_URL = "http://192.168.1.121:9011/predict"
OCR_MIN_CONFIDENCE = 0.6

DET_MODEL_DIR = (
    "/mnt/Software/200-Apps/imageFlow/models/whl/det/en/en_PP-OCRv3_det_infer/"
)
REC_MODEL_DIR = (
    "/mnt/Software/200-Apps/imageFlow/models/whl/rec/en/en_PP-OCRv3_rec_infer/"
)
CLS_MODEL_DIR = (
    "/mnt/Software/200-Apps/imageFlow/models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/"
)
USE_ANCLE_CLS = True
IMG_OCR_LANG = "en"

# --------------------------- Image OBJ ID --------------------------
OBJ_ID_HEADER = {"content-type": "application/json"}
OBJ_ID_API_URL = "http://192.168.1.121:9998/predictions"
OBJ_ID__MODEL_NAME = "yolox-x"
OBJ_ID__MIN_CONFIDENCE = 0.6


RAMDISK_DIR = "/mnt/Photos/001-Process/tmp/"
RAMDISK_SIZE_MB = 512
SAVING_FRAMES_PER_SECOND = 1 / 5
SCENE_DETECT_THRESHOLD = 2
SAMPLING_STRATEGY = ["0.1", "0.5", "0.9"]
MIN_SCENES = 2

# --------------------------- Video Processor --------------------------

TRANSCRIBE_HEADER = {"content-type": "application/json"}
TRANSCRIBE_API_URL = "http://192.168.1.121:9996/predictions"
TRANSCRIBE__MODEL_NAME = "large-v2"
TRANSCRIBE__MODEL_NAME = "medium"
#["tiny", "base", "small", "medium", "large-v1", "large-v2"],

TRASCRIBE_MODEL_SIZE_THRESHOLD = 20
# NOT IMPLEMNETED
TRANSCRIBE_MIN_CONFIDENCE = 0.5

# curl http://192.168.1.121:9996/predictions -X POST -H "Content-Type: application/json"  -d '{"input": {   "audio": "http://192.168.1.121:9999/mnt/Photos/001-Process/output-audio.aac","model": "large-v2"}}' > pizza2.json

# --------------------------- TAGS ---------------------------

OCR_TAG_OPEN = "<OCR>"
OCR_TAG_CLOSE = "</OCR>"

FACES_TAG_OPEN = "<FACES>"
FACES_TAG_CLOSE = "</FACES>"

TRANSCRIBE_TAG_OPEN = "<TRANSCRIBE>"
TRANSCRIBE_TAG_CLOSE = "</TRANSCRIBE>"

CAPTION_TAG_OPEN = "<CAPTION>"
CAPTION_TAG_CLOSE = "</CAPTION>"

OBJECTS_TAG_OPEN = "<OBJECTS>"
OBJECTS_TAG_CLOSE = "</OBJECTS>"

COLORS_TAG_OPEN = "<COLORS>"
COLORS_TAG_CLOSE = "</COLORS>"


ALLOWED_TAGS = ['OCR', 'FACES', 'TRANSCRIBE', 'CAPTION', 'OBJECTS', 'COLORS']

# --------------------------- Solr config ---------------------------
SOLR_POST_EXE = "/root/solr-8.11.1/bin/post"
SOLR_DOC_COLLECTION = "documents"
SOLR_IMG_COLLECTION = "media"
SOLR_SERVER = "http://127.0.0.1:8983/solr/"
SOLR_QUEUE = "/mnt/imageflow/solr_queue.csv"
