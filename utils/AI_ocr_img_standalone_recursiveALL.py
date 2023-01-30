import os
import subprocess

# pip3 install paddleocr
# pip3 install paddlepaddle


# [
#     [
#         [198.0, 1424.0],
#         [598.0, 1424.0],
#         [598.0, 1477.0],
#         [198.0, 1477.0],
#     ],
#     ("Text idenfied", 0.9584670662879944),
# ]
# path = "/mnt/Photos/001-Process/"
path = "/mnt/Photos/005-PhotoBook/2022/12/"
# path = "/mnt/Software/200-Apps/test/IN/"

DEST_SCREENSHOT_FOLDER = "/mnt/Photos/001-Screenshots"
SCREENSHOT_TAG = "Comment"
search_ext = [".png", ".jpg"]
# search_ext = [".jpg", ".JPG", ".png"]

files_scanned = 0
files_processed = 0


def walk_through_files(path, search_ext):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if (os.path.splitext(filename)[1]).lower() in search_ext:
                yield os.path.join(dirpath, filename)


from paddleocr import PaddleOCR

# ocr = PaddleOCR(
#     use_angle_cls=True, lang="en"
# )  # need to run only once to load model into memory
# The path of detection and recognition model must contain model and params files
ocr = PaddleOCR(
    det_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/det/en/en_PP-OCRv3_det_infer/",
    rec_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/rec/en/en_PP-OCRv3_rec_infer/",
    # rec_char_dict_path="",
    cls_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/",
    use_angle_cls=True,
    lang="en",
)
# Server model
# ocr = PaddleOCR(
#     det_model_dir="/mnt/Software/200-Apps/filemaster/models/whl/det/en/ch_ppocr_server_v2.0_det_infer/",
#     rec_model_dir="/mnt/Software/200-Apps/filemaster/models/whl/rec/en/ch_ppocr_server_v2.0_rec_infer/",
#     # rec_char_dict_path="",
#     cls_model_dir="/mnt/Software/200-Apps/filemaster/models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/",
#     use_angle_cls=True,
#     lang="en",
# )
overwrite_tags = "false"

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
    text = ""
    try:
        result = ocr.ocr(fname, det=True, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text = text + " " + line[1][0]
        print(text)
        if len(text):
            # Write it to ImageDescription and subject
            command = (
                "exiftool  -overwrite_original -subject="
                + "'"
                + str(text.replace("\r", "").replace("\n", "").replace("'", ""))
                + "' '"
                + str(fname)
                + "'"
            )
            print(command)
            res = os.system(command)
        else:
            print("NO text found")
    except Exception as e:
        print(e)
