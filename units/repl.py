import replicate
import os
import sys

sys.path.append("/mnt/No_Share/secrets/imageflow/")
import s


os.environ["REPLICATE_API_TOKEN"] = s.REPLICATE_API_TOKEN
print(s.REPLICATE_API_TOKEN)

prompt = "Please generate a vivid description of the attached image, including the emotions and feelings it evokes. Additionally, provide a commentary on the photographic style employed and the overall quality of the image. Feel free to explore the composition, lighting, colors, and any other relevant aspects that contribute to the image's impact. Your description should capture both the tangible elements of the scene and the intangible emotions it conveys, while also critically assessing the technical and artistic prowess demonstrated in the photograph."
img = "/mnt/Photos/005-PhotoBook/2023/07/_DSC3867.jpg"
# img = "https://cdn.pixabay.com/photo/2016/01/08/11/57/butterflies-1127666_640.jpg"

print(img)
# output = replicate.run(
#     "joehoover/mplug-owl:51a43c9d00dfd92276b2511b509fcb3ad82e221f6a9e5806c54e69803e291d6b",
#     input={"prompt": prompt, "img": open(img, "rb")},
# )
# The joehoover/mplug-owl model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
# o = ""
# for item in output:
#     # https://replicate.com/joehoover/mplug-owl/versions/51a43c9d00dfd92276b2511b509fcb3ad82e221f6a9e5806c54e69803e291d6b/api#output-schema
#     o = o + "" + item

# print(o)

output = replicate.run(
    "daanelson/minigpt-4:b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423",
    input={
        "prompt": prompt,
        "image": open(img, "rb"),
        "num_beams": 5,
        "temperature": 1.32,
        "top_p": 0.9,
        "repetition_penalty": 1,
        "max_new_tokens": 3000,
        "max_length": 4000,
    },
)
print(output)
