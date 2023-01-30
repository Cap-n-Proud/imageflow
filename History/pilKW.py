from PIL import Image
from PIL.ExifTags import TAGS


def add_keyword(file_path, keyword):
    with Image.open(file_path) as image:
        for k, v in image._getexif().items():
            print(TAGS.get(k, k), v)
        # print(image.info["exif"])
        # exif = image.info["exif"]
        if "keywords" in image.info:

            keywords = image.info["keywords"]
            print("AAAAAAAAAAAAAAAAAAAAA", keyword, keywords)
            if keyword not in keywords:
                keywords.append(keyword)
                image.save(file_path, "JPEG", keywords=keywords)
                print(f"Added keyword '{keyword}' to {file_path}")
            else:
                print(f"Keyword '{keyword}' already present in {file_path}")
        else:
            image.save(file_path, "JPEG", keywords=[keyword])
            print(f"Added keyword '{keyword}' to {file_path}")


if __name__ == "__main__":
    file_path = "/home/paolo/test/2023-01/IMG_1981.jpg"
    keyword = "example_keyword"
    add_keyword(file_path, keyword)
