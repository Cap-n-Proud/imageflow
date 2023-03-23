import os
import subprocess
import ProcessMedia
from PIL import Image
import face_recognition
import datetime


# path = "/mnt/Photos/001-Process/"
path = "/mnt/Photos/005-PhotoBook/2021/"
# path = "/mnt/Software/200-Apps/test/IN/"
UNKNOWN_FACE_FOLDER = "/mnt/Photos/990-Faces/unknown_faces/"
DEST_SCREENSHOT_FOLDER = "/mnt/Photos/001-Screenshots"
SCREENSHOT_TAG = "Comment"
search_ext = [".png", ".jpg", ".tif"]
# search_ext = [".jpg", ".JPG", ".png"]

files_scanned = 0
files_processed = 0


def saveFaces(filename):
    # saves all the faces in an image

    # Load the jpg file into a numpy array
    # image = face_recognition.load_image_file("biden.jpg")

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    image = face_recognition.load_image_file(fname)
    face_locations = face_recognition.face_locations(image)

    print(
        f"|saveFaces| I found {len(face_locations)} face(s) in this photograph.")

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print(
            f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d%H%M%S%f")
        print(
            f"{UNKNOWN_FACE_FOLDER}{timestamp_str}{os.path.splitext(fname)[1].lower()}")
        pil_image.save(
            f"{UNKNOWN_FACE_FOLDER}{timestamp_str}{os.path.splitext(fname)[1].lower()}")


def walk_through_files(path, search_ext):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if (os.path.splitext(filename)[1]).lower() in search_ext:
                yield os.path.join(dirpath, filename)


fpath = list(walk_through_files(path, search_ext))
total_images = str(len(list(fpath)))
print("---------------------------------------------- " + total_images)
# print(path, search_ext, list(fpath))

current_image = 0
for fname in fpath:
    current_image = current_image + 1
    print("")
    print(
        str(current_image)
        + "/"
        + str(total_images)
        + "----------------------------------------------",
        fname,
    )
    try:

        saveFaces(fname)
    except Exception as e:
        print(e)
