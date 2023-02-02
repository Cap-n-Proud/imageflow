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

# TODO: this should not be impoted here, we should use ther args
from config import fm_config

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

ProcessMedia_version = "2.0"


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

    async def ocr_image(self, fname):
        self.logger.debug("|OCR image| Starting OCR: " + str(fname))
        text = ""

        try:
            payload = '{"image":"' + str(fname) + '" ,"confidence": "0.6"}'
            self.logger.debug("|OCR Image| Payload: " + str(payload))

            r = requests.post(
                fm_config.OCR_API_URL,
                headers=fm_config.OCR_CAPTION_HEADER,
                data=payload,
            ).content

            ocr = json.loads(r.decode("utf-8"))
            self.logger.info("|OCR Image| Text: " + str(ocr["full_text"]))

            command = (
                "exiftool -overwrite_original -Caption-Abstract='"
                + str(ocr["full_text"])
                + "' '"
                + str(fname)
                + "'"
            )
            res = os.system(command)

        except Exception as e:
            self.logger.error("|OCR Image| ERROR: " + str(e))

    async def caption_image(self, fname, writeTags):
        self.logger.info("|Caption Image| Generating caption for: " + str(fname))
        try:
            payload = (
                '{"input": {"image":"'
                + str(fm_config.IMAGES_SERVER_URL)
                + str(fname)
                + '" ,"reward": "clips_grammar"}}'
            )
            self.logger.debug("|Caption Image| Payload: " + str(payload))
            self.logger.debug(
                "|Caption Image| Image server: " + str(fm_config.CAPTION_API_URL)
            )
            self.logger.debug("|Caption Image| Caption payload: " + str(payload))

            r = requests.post(
                fm_config.CAPTION_API_URL,
                headers=fm_config.IMAGE_CAPTION_HEADER,
                data=payload,
            ).content

            caption = json.loads(r.decode("utf-8"))
            if writeTags:
                command = (
                    "exiftool -overwrite_original -Caption-Abstract='"
                    + str(caption["output"])
                    + "' '"
                    + str(fname)
                    + "'"
                )
                res = os.system(command)

            self.logger.info("|Caption Image| Caption generated: " + caption["output"])
        except Exception as e:
            self.logger.warning(
                "|Caption image| Image captioning unsuccessful: "
                + str(e)
                + " "
                + str(fname)
            )
        return caption["output"]

    async def tag_image(self, fname):
        self.logger.info("|Tag Image| Image tagging started: " + str(fname))
        with open(fname, "rb") as f:
            file_bytes = f.read()

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
            command = "exiftool -overwrite_original " + tags + " '" + str(fname) + "'"
            res = os.system(command)
            self.logger.debug("|Tag Image| Tags are assigend  " + str(fname))

        return tags

    def tag_movie(self, fname):
        # with open(fname, "rb") as f:
        #     file_bytes = f.read()

        self.logger.info("Movie tags are assigend here")

    async def reverse_geotag(self, fname):
        if self.imgHasGPS(fname):
            self.logger.info("|Reverse Geocode| Reverse geocoding: " + str(fname))
            # The output variable stores the output of the  command
            command = "exiftool -c '%.9f' -GPSPosition '" + str(fname) + "'"
            self.logger.debug("|Reverse Geocode| Extracting GPS info: " + str(command))
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
                    "exiftool -overwrite_original " + command + " '" + str(fname) + "'"
                )

                res = os.system(command)
                self.logger.info(
                    "|Reverse Geocode| Reverse geocoding successfull: " + str(command)
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

    async def classify_faces(self, fname):
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
                    pix = os.listdir(fm_config.FACE_CLASSIFIER_TRAIN_DIR + str(person))
                    # Loop through each training image for the current person
                    for person_img in pix:
                        # Get the face encodings for the face in each image file
                        face = face_recognition.load_image_file(
                            fm_config.FACE_CLASSIFIER_TRAIN_DIR
                            + person
                            + "/"
                            + person_img
                        )
                        face_bounding_boxes = face_recognition.face_locations(face)

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
            names = ""
            command = ""
            for i in range(no):
                image_enc = face_recognition.face_encodings(image)[i]
                name = clf.predict([image_enc])
                names = names + " " + str(name)
                # if name == fm_config.UNKNOWN_FACE_NAME:
                #     # UNKNOWN_FACE_FOLDER:
                #     1 = 1
                #     # Code here to save the unknown face to a directory

                faces = (
                    faces
                    + ' -keywords-="no people" -keywords-='
                    + str(name[0])
                    + " -keywords+="
                    + str(name[0])
                )
            command = "exiftool -overwrite_original " + faces + " '" + str(fname) + "'"
            res = os.system(command)
            self.logger.info(
                "|Classify Faces| " + str(names) + " added as keywords to " + str(fname)
            )
        else:
            command = (
                "exiftool -overwrite_original -keywords-='no_people' -keywords+='no_people' '"
                + str(fname)
                + "'"
            )
            res = os.system(command)

    async def copy_tags_to_IPTC(self, fname):
        try:
            command = (
                'exiftool -overwrite_original "-xmp:subject<iptc:keywords" '
                + " '"
                + str(fname)
                + "'"
            )
            os.system(command)
            self.logger.info("|Copy to IPTC| Tags copied to IPTC: " + str(fname))
        except Exception as e:
            self.logger.warning(
                "|Copy to IPTC| Copy tags unsuccessful: " + str(e) + " " + str(fname)
            )

    def find_FileModifyDate(self, fname):
        command_output = str(
            subprocess.check_output(
                ["exiftool", "-FileModifyDate", str(fname), "-d", "%Y/%m"]
            )
        )

        command_output = command_output.replace("\\n", "")
        date = command_output[len(command_output) - 8 : len(command_output) - 1]
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
        f = str(os.path.dirname(fname)) + "/" + os.path.basename(fname) + ".json"
        f = fm_config.JSON_FOLDER + os.path.basename(fname) + ".json"
        # print(f)
        os.chmod(f, 0o777)
        return f

    def index_media(self, collection, fname):
        command = ""
        # Add document to search engine
        command = (
            fm_config.SOLR_POST_EXE + " -c " + collection + " '" + str(fname) + "'"
        )
        os.system(command)

    def delete_media_from_index(self, collection, fname):
        command = ""
        # http://localhost:8983/solr/MyCore/update?stream.body=

    # <delete><query>Id:298253</query></delete>&commit=true

    async def move_file2(self, file_path, dest_base_folder):
        import os
        import datetime

        # get the date the picture was taken
        date_taken = datetime.datetime.fromtimestamp(os.path.getctime(file_path))

        # create the destination folder if it doesn't exist
        year = str(date_taken.year)
        month = str(date_taken.month).zfill(2)
        dest_folder = os.path.join(dest_base_folder, year, month)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            os.chmod(dest_folder, 0o777)

        # move the file to the destination folder and update the file's timestamp
        dest_path = os.path.join(dest_folder, os.path.basename(file_path))
        os.rename(file_path, dest_path)
        # os.chmod(dest_path, 0o777)
        os.utime(dest_path, None)
        self.logger.info(f"|Move File| Moved {file_path} to {dest_path}")

    # function that moves file to a directory based on its creation date
    async def move_fileOLD(self, file_path, destination):
        from datetime import datetime
        from pathlib import Path

        # code to get file's creation date
        # code to create directory based on creation date, if it doesn't exist
        # code to move file to the directory
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime("%Y-%m")
        dst_path = Path(destination) / creation_date
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_path = dst_path / Path(file_path).name
        os.rename(file_path, dst_path)
        self.logger.info(f"|Move File| Moved {file_path} to {dst_path}")

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
                    "I found {} face(s) in {}".format(len(face_locations), str(fname))
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
