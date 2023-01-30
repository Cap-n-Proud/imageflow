# exiftool -XMP:GPSLongitude="8.574466325249867"  -XMP:GPSLatitude="47.38729870588053"  -GPSLongitudeRef="North" -GPSLatitudeRef="East" /mnt/Software/200-Apps/test/IN/IMG_1981.jpg
import pyexiv2

img = pyexiv2.Image(r"/mnt/Software/200-Apps/test/IN/202301174063.jpg")
data = img.read_exif()

print(data)
if "Exif.GPSInfo.GPSLatitude" in data:
    print("This image has GPS information.")
else:
    print("This image does not have GPS information.")
