from config import fm_config

import os
import glob
from PIL import Image
import librosa
import tempfile

# Define the directory path you want to search for files in
directory_path = "/path/to/directory"

# Define the size limit for audio files in megabytes
size_limit_mb = 10

# Define a string to store all transcriptions
transcription_string = ""

# Loop through all files in the directory
for file_path in glob.glob(os.path.join(directory_path, "*")):
    # Check if the file is an image
    if file_path.endswith(".jpg") or file_path.endswith(".png"):
        # Open the image file
        image = Image.open(file_path)

        # Call your specific image processing functions here
        # Example:
        # processed_image = process_image(image)

        # Save the processed image to a new file
        # Example:
        # processed_image.save(os.path.join(directory_path, "processed_" + os.path.basename(file_path)))

    # Check if the file is audio
    elif file_path.endswith(".mp3") or file_path.endswith(".wav"):
        # Get the file size in bytes and convert to megabytes
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb > size_limit_mb:
            # Split the audio file into chunks of the specified size limit
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            chunk_size = int(size_limit_mb * 1024 * 1024 / (4 * sample_rate))
            chunks = librosa.util.frame(
                audio_data, frame_length=chunk_size, hop_length=chunk_size)

            # Loop through each chunk and transcribe it
            for i, chunk in enumerate(chunks.T):
                # Create a temporary file on a RAM disk and write the audio data to it
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpfile_path = os.path.join(
                        tmpdir, "audio_chunk_{}.wav".format(i))
                    librosa.output.write_wav(tmpfile_path, chunk, sample_rate)

                    # Call your transcription function with the temporary file path
                    # Example:
                    # transcription = transcribe_audio(tmpfile_path)
                    payload = (
                        f'{{"input": {{"audio":"{fm_config.IMAGES_SERVER_URL}{tmpfile_path}", "model":"tiny"}}}}'
                    )

                    print(
                        f"|Transcribe| Payload: '{str(payload)}'")

                    r = requests.post(
                        fm_config.TRANSCRIBE_API_URL,
                        headers=fm_config.TRANSCRIBE_HEADER,
                        data=payload,
                        timeout=60 * 60 * 3
                    ).text

                    print(
                        f"|Transcribe| Transcribe success, result: {str(r)}")
                    # Append the transcription to the transcription string
                    # Example:
                    transcription_string += transcription

        else:
            # Load the audio data
            audio_data, sample_rate = librosa.load(file_path, sr=None)

            # Create a temporary file on a RAM disk and write the audio data to it
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile_path = os.path.join(tmpdir, "audio.wav")
                librosa.output.write_wav(tmpfile_path, audio_data, sample_rate)

                # Call your transcription function with the temporary file path
                # Example:
                # transcription = transcribe_audio(tmpfile_path)

                # Append the transcription to the transcription string
                # Example:
                # transcription_string += transcription

        # Save the transcription string to a new file
        # Example:
        # with open(os.path.join(directory_path, "transcription_" + os.path.basename(file_path) + ".txt"), "w") as f:
        #     f.write(transcription_string)
        print(transcription_string)
    # If the file is not an image or audio file, skip it
    else:
        continue
