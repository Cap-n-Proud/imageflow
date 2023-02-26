import cv2
import numpy as np
import base64


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


import cv2
import numpy as np


def resize_to_high_res(image_path, max_width=1920):
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

    # cv2.imwrite("./out.jpg", img_resized)

    # Encode the resized image as a JPEG bytes object
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_bytes = cv2.imencode(".jpg", img_resized, encode_param)

    return img_bytes.tobytes()


async def resize_with_aspect_ratio(
    file_path, max_width=None, max_height=None, format=None
):
    # Load the image from the file using async I/O
    file_bytes = read_file(file_path)
    max_width = int(max_width)
    # max_height = int(max_height)

    # Decode the file contents from base64 and into a NumPy array
    nparr = np.frombuffer(base64.b64decode(file_bytes), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check if the image is None
    if image is None:
        print("Failed to decode image")
    size_bytes = len(image)
    # Convert the size to megabytes
    size_mb = size_bytes / (1024 ** 2)

    print(f"The size of the image is {size_mb:.2f} megabytes")

    # Get the original width and height of the image
    orig_height, orig_width = image.shape[:2]

    # Compute the aspect ratio of the original image
    aspect_ratio = orig_width / orig_height

    # Compute the target width and height based on the maximum dimensions
    if max_width is not None and max_height is not None:
        # Both max width and max height are specified
        target_width = max_width
        target_height = max_height
    elif max_width is not None:
        # Only max width is specified
        target_width = max_width
        target_height = int(target_width / aspect_ratio)
    elif max_height is not None:
        # Only max height is specified
        target_height = max_height
        target_width = int(target_height * aspect_ratio)
    else:
        # Neither max width nor max height is specified, return original image
        return resources_pb2.Image(base64=file_bytes)

    # Resize the image using OpenCV
    resized_image = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Convert the resized image to RGB format
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./", rgb_image)

    # Encode the resized image to the specified format
    if format is None:
        ext = file_path.lower().split(".")[-1]
    else:
        ext = format.lower()
    if ext == "jpg" or ext == "jpeg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    elif ext == "png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    elif ext == "webp":
        encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), 90]
    else:
        raise ValueError(f"Unsupported image format: {ext}")
    success, encoded_image = cv2.imencode("." + ext, rgb_image, encode_params)
    if not success:
        raise RuntimeError("Failed to encode image")

    # Convert the encoded image to a base64-encoded string and wrap it in an Image protobuf message
    encoded_string = base64.b64encode(encoded_image).decode("utf-8")
    return resources_pb2.Image(base64=encoded_string)


image_path = "in.jpg"
high_res = (1920, 1080)
# resized_image = resize_with_aspect_ratio(image_path, "./out.jpg", max_width=1920)
# resized_image = resize_with_aspect_ratio(
#     image_path, max_width=1920, max_height=None, format="jpg"
# )

resized_image = resize_to_high_res(image_path)
