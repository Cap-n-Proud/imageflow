from paddleocr import PaddleOCR
import sys

sys.path.append("/mnt/Software/200-Apps/imageFlow")
from config import fm_config

fname = "/mnt/Software/200-Apps/test/IN/download.jpg"

#
# ocr = PaddleOCR(
#     det_model_dir=fm_config.DET_MODEL_DIR,
#     rec_model_dir=fm_config.REC_MODEL_DIR,
#     # rec_char_dict_path="",
#     cls_model_dir=fm_config.CLS_MODEL_DIR,
#     use_angle_cls=fm_config.USE_ANCLE_CLS,
#     lang=fm_config.IMG_OCR_LANG,
# )

ocr = PaddleOCR(
    det_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/det/en/en_PP-OCRv3_det_infer/",
    rec_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/rec/en/en_PP-OCRv3_rec_infer/",
    # rec_char_dict_path="",
    cls_model_dir="/mnt/Software/200-Apps/filemaster_old/models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/",
    use_angle_cls=True,
    lang="en",
)
text = ""
try:
    result = ocr.ocr(fname, det=True, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = text + " " + line[1][0]
    if len(text):
        print(text)

    else:
        print("NO text found")
except Exception as e:
    print(e)
