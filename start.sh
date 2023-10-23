# Process images in the InstantUpload folder
python3 /app/controller.py --moveFileImage True -id "/mnt/Photos/005-PhotoBook/" -iw "/mnt/Photos/000-InstantUpload" -s /mnt/secrets/imageflow -c /mnt/imageflow/config -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

# Produced Videos are proecessed but not moved
python3 /app/controller.py --moveFileImage False -id "/mnt/Photos/010-Videos/000-Produced/" -iw "/mnt/Photos/010-Videos/000-Produced/" -s /mnt/secrets/imageflow -c /mnt/imageflow/config -fc "/mnt/Apps_Config/imageflow/faceClassifier.pkl"

sleep infinity
