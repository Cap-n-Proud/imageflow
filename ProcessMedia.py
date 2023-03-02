import os
import random
import requests
import subprocess
import re
import json
import face_recognition
from PIL import Image
import logging
import pickle as cPickle
from sklearn import svm
import sys
import pyexiv2
import base64
from pathlib import Path

from datetime import timedelta
import cv2
import numpy as np
from pathlib import Path

# TODO: this should not be impoted here, we should use ther args
from config import fm_config

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images

ProcessMedia_version = "4.0"


class ProcessMedia:
    logger = None

    async def info(self):
        print(ProcessMedia_version)

    def __init__(self, args):
        ProcessMedia.logger = logger
        self.args = args
        sys.path.append(self.args.secretsPath)
        import s

        self.s = s

        # print(s.REVERSE_GEO_API)
        pass

    def createTempDirectory(self, fname):

        directory = os.path.splitext(os.path.basename(fname))[0]
        directory = directory.replace("'", "")
        # Check if the directory exists already
        if os.path.isdir(fm_config.RAMDISK_DIR + directory):
            self.logger.debug(f"Directory {directory} already exists.")
            return
        command = f"mkdir '{fm_config.RAMDISK_DIR}{directory}'"
        os.system(command)
        self.logger.debug(
            f"Directory created at '{fm_config.RAMDISK_DIR}{directory}'")

    def removeTempDirectory(self, fname):
        directory = os.path.splitext(os.path.basename(fname))[0]
        directory = directory.replace("'", "")
        # Check if the directory exists
        if not os.path.isdir(fm_config.RAMDISK_DIR + directory):
            self.logger.info(
                f"Directory {fm_config.RAMDISK_DIR}{directory} does not exists."
            )
            return
        os.system(f"rm -r '{fm_config.RAMDISK_DIR}{directory}'")
        self.logger.info(
            f"Directory removed: '{fm_config.RAMDISK_DIR}{directory}'.")

    def create_ramdisk(self, directory, size):
        # Check if the directory exists already
        if os.path.isdir(directory):
            self.logger.debug(f"Directory {directory} already exists.")
            return

        os.system(f"mkdir {directory}")

        # Create the ramdisk
        # os.system(f"sudo mount -t tmpfs -o size={size}M tmpfs {directory}")
        print(f"Created ramdisk at {directory} with size {size}MB.")

    def remove_ramdisk(self, directory):
        # Check if the directory exists already
        if not os.path.isdir(directory):
            print(f"Directory {directory} does not exists.")
            return

        # Remove the ramdisk
        import shutil

        shutil.rmtree(directory)
        print(f"Ramdisk content at {directory} cleared")

    def split_video_into_scenes(self, video_path, output_dir=None, threshold=1):
        # Open our video, create a scene manager, and add a detector.
        self.logger.debug(
            f"Finding scenes for {video_path}, threshold {threshold}")
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(adaptive_threshold=threshold))
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        filename, file_extension = os.path.splitext(
            os.path.basename(video_path))
        self.logger.debug(f"Scenes {len(scene_list)}.")
        output_dir = output_dir + "/" + filename
        if len(scene_list) > fm_config.MIN_SCENES:
            save_images(
                scene_list=scene_list, video=video, output_dir=output_dir, num_images=1
            )
        else:
            a = " "
            self.logger.info(
                f"Limited number of scenes found: {video_path}, threshold {threshold}. Proceeding with time-fixed sampling strategy: {fm_config.SAMPLING_STRATEGY}")
            video = cv2.VideoCapture(video_path)
            sampling_strategy = fm_config.SAMPLING_STRATEGY
            screenshot_number = 0
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            result = [int(frame_count * float(sample))
                      for sample in sampling_strategy]
            sampling_frames = ",".join(map("{:,}".format, result))
            self.logger.info(
                f"frames per second: {fps}, frame count: {frame_count}, sampling frames: {a.join(sampling_frames.split(','))}"
            )

            for sample in sampling_strategy:
                frame_id = int(frame_count * float(sample))
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video.read()
                cv2.imwrite(
                    f"{output_dir}/screenshot{screenshot_number}.jpg", frame)
                screenshot_number += 1

    def extract_audio(self, fname, tmp_folder):
        filename, file_extension = os.path.splitext(os.path.basename(fname))
        filename = filename.replace("'", "")

        command = f"ffmpeg -y -i '{fname}' -vn -acodec copy '{tmp_folder}{filename}/audio.aac'"
        self.logger.debug(f"|extract_audio|: {command}")
        subprocess.call(command, shell=True)

    def imgHasGPS(self, fname):

        img = pyexiv2.Image(str(fname))
        data = img.read_exif()
        img.close()
        if "Exif.GPSInfo.GPSLatitude" in data:
            self.logger.debug("image has GPS information.")
            return True

        else:
            self.logger.debug("image has NO GPS information")

            return False

    async def resize_to_high_res(self, image_path, max_width=1980):
        """
        Resizes an image to a specified high resolution.

        :param image_path: The path to the input image.
        :param high_res: The desired high resolution size in pixels, as a tuple of (width, height).
        :return: The resized image as a bytes object.
        """
        # Load the image and resize it to the high resolution
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Get the original width and height of the image
        # Compute the aspect ratio of the original image
        orig_height, orig_width = img.shape[:2]
        aspect_ratio = orig_width / orig_height
        target_width = max_width
        target_height = int(target_width / aspect_ratio)
        # Resize the image using OpenCV
        img_resized = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )
        resized_height, resized_width = img_resized.shape[:2]

        # cv2.imwrite(
        #     str(os.path.dirname(os.path.abspath(image_path))) + "/out.jpg", img_resized
        # )
        # print(str(os.path.dirname(os.path.abspath(image_path))) + "/out.jpg")

        # Encode the resized image as a JPEG bytes object
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, img_bytes = cv2.imencode(".jpg", img_resized, encode_param)
        self.logger.info(
            "|Tag Image| Image resized: "
            + str(image_path)
            + " "
            + str(resized_height)
            + " x "
            + str(resized_width)
        )

        return img_bytes.tobytes()

    async def read_file(self, path):
        with open(path, "rb") as f:
            return f.read()

    import re

    async def stripIt(self, s):
        """
        Removes all html tags from a string, leaving just the
        content behind.
        """
        a = re.sub('<.*?>', '', s).replace('"', '').replace("\n", "")

        return re.sub('\s{2,}', ' ', a)

    async def write_keywords_metadata_to_video_file(
        self, video_file_path, keywords, description
    ):
        import pathlib

        """
        Writes keywords metadata to a QuickTime video file, removing duplicates.

        Parameters:
        video_file_path (str): The path to the video file.
        keywords (list): A list of keywords to write as metadata.

        Returns:
        None
        """

        # First, we need to get the current metadata for the file using ffprobe
        ffprobe_cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_file_path}"'
        ffprobe_output = subprocess.check_output(ffprobe_cmd, shell=True).decode(
            "utf-8"
        )
        metadata = json.loads(ffprobe_output)["format"]["tags"]

        # Get the existing keywords from the metadata and remove duplicates
        existing_keywords = set(metadata.get("keywords", "").split(","))
        keywords = list(set(keywords) - existing_keywords)

        # If there are no new keywords to add, we don't need to do anything
        # if not keywords:
        #     return
        ffmpeg_cmd = ""
        # Add the new keywords to the metadata dictionary
        all_keywords = existing_keywords.union(set(keywords))
        metadata["keywords"] = ",".join(all_keywords)

        # If the keywords tag didn't exist in the metadata, add it now
        if "keywords" not in metadata:
            metadata["keywords"] = ",".join(keywords)

        # Finally, we use ffmpeg to write the new metadata back to the file
        ffmpeg_cmd = f'ffmpeg -i "{video_file_path}" '

        for key, value in metadata.items():
            ffmpeg_cmd += f"-metadata {key}='{value}' "

        d = await self.stripIt(description)

        ffmpeg_cmd += f'-metadata description="{d}" '

        new_file = f"{pathlib.Path(video_file_path).parent}/{pathlib.Path(video_file_path).stem}_new{pathlib.Path(video_file_path).suffix}"
        ffmpeg_cmd += f'-c copy "{new_file}"'
        self.logger.debug(
            f"|write_keywords_metadata_to_video_file|: {ffmpeg_cmd}")
        try:
            command_run = subprocess.call(ffmpeg_cmd, shell=True)
            if command_run == 0:
                self.logger.info(
                    f"|write_keywords_metadata_to_video_file| keywords successfully written: {ffmpeg_cmd}")
                os.remove(video_file_path)
                os.rename(f"{new_file}", video_file_path)
            else:
                self.logger.error(
                    f"|write_keywords_metadata_to_video_file| The following command generated an error: {ffmpeg_cmd}")

            # If everything worked correctly, we can delete the old file and rename the new one

        except Exception as e:
            self.logger.error(
                f"|write_keywords_metadata_to_video_file| Error: {e}")

    async def writeTagsMedia(self, fname, KW=None, caption=None, description=None):

        keyCount = 0
        keyNames = ""
        keyword = ""
        for key in KW:
            keyCount += 1
            keyNames = keyNames + str(key) + " "
            keyword = keyword + f" -keywords-='{key}' -keywords+='{key}'"
        command = f"exiftool -overwrite_original {keyword} '{str(fname)}'"
        self.logger.info(f"|Tag Media| Command: {command}")

        res = os.system(command)
        self.logger.info(
            f"|Tag Media| {keyCount} keywords added: {keyNames} to '{str(fname)}'"
        )

    async def id_obj_image(self, fname, writeTags):
        import ast

        self.logger.debug(
            f"|OBJ ID Image| Starting identification: '{str(fname)}'")
        text = ""

        try:
            payload = (
                '{"input":{"image":"'
                + str(fm_config.IMAGES_SERVER_URL)
                + str(fname)
                + '","confidence":"'
                + str(fm_config.OBJ_ID__MIN_CONFIDENCE)
                + '","model_name":"'
                + str(fm_config.OBJ_ID__MODEL_NAME)
                + '"}}'
            )
            self.logger.debug(
                f"|OBJ ID Image| Obj ID details: {fm_config.OBJ_ID_API_URL}, {fm_config.OBJ_ID_HEADER}"
            )
            self.logger.debug(f"|OBJ ID Image| Payload: {str(payload)}")

            url = fm_config.OBJ_ID_API_URL
            headers = fm_config.OBJ_ID_HEADER
            data = "'" + payload + "'"
            data = payload
            # print("curl ", url, " ", headers, "-d ", data)
            response = requests.post(url, headers=headers, data=data)
            response = response.text

            data = json.loads(response)

            inference = json.loads(
                data["output"]["inference"].replace("'", '"'))
            unique_cls = list(set([d["cls"] for d in inference]))

            self.logger.info(f"|OBJ ID Image| Text: {str(unique_cls)}")
            clsCount = 0
            tags = ""
            tagNames = ""
            if writeTags:
                for cls in unique_cls:
                    clsCount += 1
                    tagNames = tagNames + str(cls) + " "
                    tags = (
                        tags
                        + " -keywords-='"
                        + cls
                        + "' "
                        + " -keywords+='"
                        + cls
                        + "' "
                    )
                command = (
                    "exiftool -overwrite_original " +
                    tags + " '" + str(fname) + "'"
                )
                res = os.system(command)
                self.logger.info(
                    f"|Tag Image| {clsCount} objects identified {tagNames} to {str(fname)}"
                )
            else:
                return str(unique_cls)

        except Exception as e:
            self.logger.error(f"|OBJ ID Image| ERROR: {str(e)}")

    async def ocr_image(self, fname, writeTags):
        self.logger.debug(f"|OCR image| Starting OCR: {str(fname)}")
        text = ""

        try:
            payload = (
                '{"image":"'
                + str(fname)
                + '" ,"confidence":"'
                + str(fm_config.OCR_MIN_CONFIDENCE)
                + '"}'
            )
            self.logger.debug(f"|OCR Image| Payload: {str(payload)}")

            r = requests.post(
                fm_config.OCR_API_URL,
                headers=fm_config.OCR_CAPTION_HEADER,
                data=payload,
            ).content

            ocr = json.loads(r.decode("utf-8"))
            # self.logger.info(f"|OCR Image| Text: {str(ocr['full_text'])}")
            self.logger.info(
                f'|OCR Image| successfull: {str(ocr["full_text"])}')
            if writeTags:
                command = (
                    "exiftool -overwrite_original -Caption-Abstract='"
                    + str(ocr["full_text"])
                    + "' '"
                    + str(fname)
                    + "'"
                )
                res = os.system(command)
            else:
                return str(ocr["full_text"])

        except Exception as e:
            self.logger.error(f"|OCR Image| ERROR: {str(e)}")

    async def preProcessVideo(self, fname):
        self.createTempDirectory(fname)
        self.split_video_into_scenes(
            fname,
            threshold=fm_config.SCENE_DETECT_THRESHOLD,
            output_dir=fm_config.RAMDISK_DIR,
        )

        self.extract_audio(fname, fm_config.RAMDISK_DIR)

    async def transcribe(self, fname):
        # -d '{"input": {   "audio": "http://192.168.1.121:9999/mnt/Photos/001-Process/audio.aac","model": "large-v2"}}'

        # Check file size if too big, we use a smaller models to avoid timeout
        fileSize = os.path.getsize(fname) / 1048576
        print(os.path.getsize(fname))
        if fileSize > fm_config.TRASCRIBE_MODEL_SIZE_THRESHOLD:
            model = "small"
        else:
            model = fm_config.TRANSCRIBE__MODEL_NAME

        self.logger.info(
            f"|Transcribe| Generating transcription for: '{str(fname)}' ({round(fileSize,2)} MB). Model: '{model}'")

        try:
            # payload = f'{"input": {"audio": "{str(fm_config.IMAGES_SERVER_URL)}{str(fname)}", "model": "{str(model)}"}}'

            payload = (
                '{"input": {"audio":"'
                + str(fm_config.IMAGES_SERVER_URL)
                + str(fname)
                + '" ,"model":"'
                + str(model)
                + '"}}'
            )
            self.logger.debug(f"|Transcribe| Payload: '{str(payload)}'")
            self.logger.debug(
                f"|Transcribe| Transcribe server: '{str(fm_config.TRANSCRIBE_API_URL)}'"
            )

            r = requests.post(
                fm_config.TRANSCRIBE_API_URL,
                headers=fm_config.TRANSCRIBE_HEADER,
                data=payload,
                timeout=60 * 60 * 3
            ).text

            self.logger.debug(
                f"|Transcribe| Transcribe success, result: {str(r)}")

            transcription = json.loads(r)
            # transcription = json.loads(r.decode("utf-8"))
            # self.logger.info(f"|Transcribe| Transcription generated for file {str(fname)}: {transcription["output"]["transcription"]})
            self.logger.info(
                "|Transcribe| Transcription generated for file: "
                + str(fname)
                + ": "
                + transcription["output"]["transcription"][:100]
            )

        except Exception as e:
            self.logger.error(
                "|Transcription| Transcription unsuccessful: "
                + str(e)
                + " "
                + str(fname)
            )

        return str(transcription["output"]["transcription"])

    async def caption_image(self, fname, writeTags):
        self.logger.info(
            f"|Caption Image| Generating caption for: '{str(fname)}'")
        try:
            payload = (
                '{"input": {"image":"'
                + str(fm_config.IMAGES_SERVER_URL)
                + str(fname)
                + '" ,"reward": "clips_grammar"}}'
            )
            self.logger.debug(f"|Caption Image| Payload: '{str(payload)}'")
            self.logger.debug(
                f"| Caption Image | Image server: '{str(fm_config.CAPTION_API_URL)}'"
            )

            r = requests.post(
                fm_config.CAPTION_API_URL,
                headers=fm_config.IMAGE_CAPTION_HEADER,
                data=payload,
            ).content

            caption = json.loads(r.decode("utf-8"))
            self.logger.info(
                f"|Caption Image| Caption generated for file '{str(fname)}': '{caption['output']}'"
            )
            if writeTags:
                command = (
                    "exiftool -overwrite_original -Caption-Abstract='"
                    + str(caption["output"])
                    + "' '"
                    + str(fname)
                    + "'"
                )
                res = os.system(command)
            else:
                return str(caption["output"])

        except Exception as e:
            self.logger.error(
                f"|Caption image| Image captioning unsuccessful: {str(e)} '{str(fname)}'"
            )
            # resize_with_aspect_ratio(
            #     file_path, max_width=None, max_height=None, format=None
            # ):

    async def tag_image(self, fname):
        self.logger.info("|Tag Image| Image tagging started: " + str(fname))
        # with open(fname, "rb") as f:
        #     file_bytes = f.read()
        try:
            file_bytes = await self.resize_to_high_res(fname)
        # file_bytes = await self.resize_with_aspect_ratio(
        #     fname, max_width=1920, max_height=None, format="jpg"
        # )

        except:
            self.logger.warning("An exception occurred: " + str(fname))
            return
        # with open(fname, "rb") as f:
        #     file_bytes = f.read()

        channel = ClarifaiChannel.get_json_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        metadata = self.s.CLARIFAI_AUTH

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                model_id="aaa03c23b3724a16a56b629203edc62c",
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=file_bytes)
                        )
                    )
                ],
                model=resources_pb2.Model(
                    output_info=resources_pb2.OutputInfo(
                        output_config=resources_pb2.OutputConfig(
                            max_concepts=50, min_value=0.50
                        )
                    )
                ),
            ),
            metadata=metadata,
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise Exception(
                "Post model outputs failed, status: "
                + post_model_outputs_response.status.description
            )

        # Since we have one input, one output will exist here.
        output = post_model_outputs_response.outputs[0]
        tags = ""
        tagNames = ""
        tagCount = 0
        # self.logger.info("Predicted concepts:")
        for concept in output.data.concepts:
            # self.logger.info("%s %.2f" % (concept.name, concept.value))
            # self.logger.info(concept.name)
            tagCount += 1
            tagNames = tagNames + concept.name + " "
            tags = (
                tags
                + " -keywords-='"
                + concept.name
                + "' "
                + " -keywords+='"
                + concept.name
                + "' "
            )  # + " -iptc:keywords-='" + concept.name + "' " + " -iptc:keywords+='" + concept.name + "' "
        if len(tags) > 0:
            self.logger.info("|Tag Image| Tags generated: " + tagNames)
            command = "exiftool -overwrite_original " + \
                tags + " '" + str(fname) + "'"
            res = os.system(command)
            self.logger.debug("|Tag Image| Tags are assigend  " + str(fname))

        return tags

    async def reverse_geotag(self, fname):
        if self.imgHasGPS(fname):
            self.logger.info(
                "|Reverse Geocode| Reverse geocoding: " + str(fname))
            # The output variable stores the output of the  command
            command = "exiftool -c '%.9f' -GPSPosition '" + str(fname) + "'"
            self.logger.debug(
                "|Reverse Geocode| Extracting GPS info: " + str(command))
            output = subprocess.getoutput(command)
            command = ""
            try:
                temp = re.findall(r"\d+", output)
                lat = temp[0] + "." + temp[1]
                lon = temp[2] + "." + temp[3]

                url = (
                    fm_config.REVERSE_GEO_URL
                    + str(lat)
                    + ","
                    + str(lon)
                    + "&key="
                    + self.s.REVERSE_GEO_API
                )

                response = requests.get(url)

                data = json.loads(response.text)
                reverse_geo = data["results"][0]["components"]
                # print(reverse_geo["postcode"])
                for key, value in reverse_geo.items():
                    command = (
                        command
                        + " -keywords-='"
                        + str(value)
                        + "'"
                        + " -keywords+='"
                        + str(value)
                        + "'"
                    )
                command = (
                    "exiftool -overwrite_original " +
                    command + " '" + str(fname) + "'"
                )

                res = os.system(command)
                self.logger.info(
                    "|Reverse Geocode| Reverse geocoding successfull: " +
                    str(command)
                )

            except Exception as e:
                self.logger.warning(
                    "|Reverse Geocode| Reverse geocoding unsuccessful: "
                    + str(e)
                    + " "
                    + str(fname)
                )
                command = " -keywords-=no_GPS_tag  -keywords+=no_GPS_tag"
                res = os.system(command)
        else:
            self.logger.info("NO GPS info found in image")

    def saveFaces(image):
        # saves all the faces in an image

        # Load the jpg file into a numpy array
        # image = face_recognition.load_image_file("biden.jpg")

        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # See also: find_faces_in_picture_cnn.py
        face_locations = face_recognition.face_locations(image)

        print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                    top, left, bottom, right
                )
            )

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.show()

    async def classify_faces(self, fname, writeTags):
        encodings = []
        names = []

        # Load the  image with unknown faces into a numpy array
        image = face_recognition.load_image_file(fname)

        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(image)
        no = len(face_locations)
        self.logger.info(
            "|Classify Faces| Number of faces detected: "
            + str(no)
            + " in "
            + str(fname)
        )
        if no > 0:
            # Loop through each person in the training directory
            if os.path.isfile(self.args.faceClassifierFile):
                self.logger.debug(
                    "|Classify Faces| Using classifier: "
                    + str(self.args.faceClassifierFile)
                )
                with open(self.args.faceClassifierFile, "rb") as fid:
                    clf = cPickle.load(fid)
            else:
                self.logger.info(
                    "|Classify Faces| Classifier file does not exist, training  it now."
                )
                train_dir = os.listdir(fm_config.FACE_CLASSIFIER_TRAIN_DIR)
                for person in train_dir:
                    pix = os.listdir(
                        fm_config.FACE_CLASSIFIER_TRAIN_DIR + str(person))
                    # Loop through each training image for the current person
                    for person_img in pix:
                        # Get the face encodings for the face in each image file
                        face = face_recognition.load_image_file(
                            fm_config.FACE_CLASSIFIER_TRAIN_DIR
                            + person
                            + "/"
                            + person_img
                        )
                        face_bounding_boxes = face_recognition.face_locations(
                            face)

                        # If training image contains exactly one face
                        if len(face_bounding_boxes) == 1:
                            face_enc = face_recognition.face_encodings(face)[0]
                            # Add face encoding for current image with corresponding label (name) to the training data
                            encodings.append(face_enc)
                            names.append(person)
                        else:
                            command = (
                                "rm "
                                + fm_config.FACE_CLASSIFIER_TRAIN_DIR
                                + person
                                + "/"
                                + person_img
                            )
                            c = os.system(command)
                            self.logger.warning(
                                person
                                + "/"
                                + person_img
                                + " was skipped and can't be used for training, image removed."
                            )
                # Create and train the SVC classifier
                clf = svm.SVC(gamma="scale")
                clf.fit(encodings, names)
                with open(str(self.args.faceClassifierFile), "wb") as fid:
                    cPickle.dump(clf, fid)
                self.logger.info(
                    "|Classify Faces| Classifier saved: "
                    + str(self.args.faceClassifierFile)
                )
            # Predict all the faces in the test image using the trained classifier
            faces = ""
            names = []
            command = ""
            for i in range(no):
                image_enc = face_recognition.face_encodings(image)[i]
                name = clf.predict([image_enc])
                # names = names + " " + str(name)
                if name[0] not in names:
                    names.append(name[0])

            if writeTags:
                faces = (
                    faces
                    + ' -keywords-="no people" -keywords-='
                    + str(name[0])
                    + " -keywords+="
                    + str(name[0])
                )
                command = (
                    "exiftool -overwrite_original " +
                    faces + " '" + str(fname) + "'"
                )
                res = os.system(command)
                self.logger.info(
                    "|Classify Faces| "
                    + str(names)
                    + " added as keywords to "
                    + str(fname)
                )
            else:
                return str(names)
        else:
            if writeTags:
                command = (
                    "exiftool -overwrite_original -keywords-='no_people' -keywords+='no_people' '"
                    + str(fname)
                    + "'"
                )
                res = os.system(command)
            else:
                return str("no_people")

    async def copy_tags_to_IPTC(self, fname):
        try:
            command = (
                'exiftool -overwrite_original "-xmp:subject<iptc:keywords" '
                + " '"
                + str(fname)
                + "'"
            )
            os.system(command)
            self.logger.info(
                "|Copy to IPTC| Tags copied to IPTC: " + str(fname))
        except Exception as e:
            self.logger.warning(
                "|Copy to IPTC| Copy tags unsuccessful: " +
                str(e) + " " + str(fname)
            )

    def find_FileModifyDate(self, fname):
        command_output = str(
            subprocess.check_output(
                ["exiftool", "-FileModifyDate", str(fname), "-d", "%Y/%m"]
            )
        )

        command_output = command_output.replace("\\n", "")
        date = command_output[len(command_output) - 8: len(command_output) - 1]
        return date

    def create_json(self, fname):
        command = (
            "exiftool -json  "
            + str(fname)
            + " > "
            + fm_config.JSON_FOLDER
            + os.path.basename(fname)
            + ".json"
        )
        os.system(command)
        f = str(os.path.dirname(fname)) + "/" + \
            os.path.basename(fname) + ".json"
        f = fm_config.JSON_FOLDER + os.path.basename(fname) + ".json"
        # print(f)
        os.chmod(f, 0o777)
        return f

    def index_media(self, collection, fname):
        command = ""
        # Add document to search engine
        command = (
            fm_config.SOLR_POST_EXE + " -c " +
            collection + " '" + str(fname) + "'"
        )
        os.system(command)

    def delete_media_from_index(self, collection, fname):
        command = ""
        # http://localhost:8983/solr/MyCore/update?stream.body=

    async def move_file(self, fname, dest_folder):
        # os.chmod(fname, 0o777)
        command = (
            "exiftool  -overwrite_original '-Directory<FileModifyDate' '-Directory<CreateDate' -d "
            + str(dest_folder)
            + "/%Y/%m/ '"
            + str(fname)
            + "'"
        )
        try:
            res = os.system(command)
            self.logger.info(
                f"|Move File| Moved " + str(fname) + " to " + str(dest_folder)
            )
        except FileNotFoundError as e:
            logger.error("File not found: " + str(row[1]) + " " + str(e))

    def crop_faces(self, source_dir, dest_dir, file_extension):
        # TODO:ADD condition     if len(face_bounding_boxes) == 1: before saving image

        def walk_through_files(path, file_extension=(".jpg")):
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in file_extension:
                        yield os.path.join(dirpath, filename)

        f = walk_through_files(str(source_dir))
        self.logger.info(
            "Starting cropping faces in " + str(len(list(f))) + " pictures"
        )
        for fname in walk_through_files(str(source_dir)):
            # Find all the faces in the image using the default HOG-based model.
            # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
            # See also: find_faces_in_picture_cnn.py
            try:
                imageArray = face_recognition.load_image_file(fname)
                face_locations = face_recognition.face_locations(imageArray)

                self.logger.info(
                    "I found {} face(s) in {}".format(
                        len(face_locations), str(fname))
                )

                for face_location in face_locations:

                    # self.logger.info the location of each face in this image
                    top, right, bottom, left = face_location
                    self.logger.info(
                        "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                            top, left, bottom, right
                        )
                    )

                    # You can access the actual face itself like this:
                    face_image = imageArray[top:bottom, left:right]
                    pil_image = Image.fromarray(face_image)
                    try:
                        fileName = (
                            str(dest_dir)
                            + re.findall("\d{4}-\d{2}-\d{2}", fname)[0]
                            + "-"
                            + str(random.randint(1000, 9999))
                            + ".jpg"
                        )
                    except:
                        fileName = (
                            str(dest_dir)
                            + "no-date"
                            + "-"
                            + str(random.randint(1000, 9999))
                            + ".jpg"
                        )
                    pil_image.save(fileName)
                    os.system("chmod 777 " + str(fileName))
                    # self.logger.info(fname, fileName)

            except:
                self.logger.error("No picture or error")

    def batch_process_media(
        self, origPath, destPath, tag_image, tag_media, reverse_geotag, move
    ):
        self.logger.info("Batch process")
