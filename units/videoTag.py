import subprocess
import json
import os
import pathlib
# ffprobe -v quiet -print_format json -show_format -show_streams "/mnt/Photos/001-Process/IN/video.mp4"

import subprocess
import json
import os
import pathlib


def write_keywords_metadata_to_video_file(video_file_path, keywords):
    """
    Writes keywords metadata to a QuickTime video file, removing duplicates.

    Parameters:
    video_file_path (str): The path to the video file.
    keywords (list): A list of keywords to write as metadata.

    Returns:
    None
    """

    # First, we need to get the current metadata for the file using ffprobe
    ffprobe_cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_file_path}"'
    ffprobe_output = subprocess.check_output(
        ffprobe_cmd, shell=True).decode('utf-8')
    metadata = json.loads(ffprobe_output)['format']['tags']

    # Get the existing keywords from the metadata and remove duplicates
    existing_keywords = set(metadata.get('keywords', '').split(','))
    keywords = list(set(keywords) - existing_keywords)

    # If there are no new keywords to add, we don't need to do anything
    if not keywords:
        return

    # Add the new keywords to the metadata dictionary
    all_keywords = existing_keywords.union(set(keywords))
    metadata['keywords'] = ','.join(all_keywords)

    # If the keywords tag didn't exist in the metadata, add it now
    if 'keywords' not in metadata:
        metadata['keywords'] = ','.join(keywords)

    # Finally, we use ffmpeg to write the new metadata back to the file
    ffmpeg_cmd = f'ffmpeg -i "{video_file_path}" '

    for key, value in metadata.items():
        ffmpeg_cmd += f'-metadata {key}="{value}" '

    new_file = f"{pathlib.Path(video_file_path).parent}/{pathlib.Path(video_file_path).stem}_new{pathlib.Path(video_file_path).suffix}"
    ffmpeg_cmd += f'-c copy "{new_file}"'

    subprocess.call(ffmpeg_cmd, shell=True)

    # If everything worked correctly, we can delete the old file and rename the new one
    os.remove(video_file_path)
    os.rename(f"{new_file}", video_file_path)


write_keywords_metadata_to_video_file(
    "/mnt/Photos/001-Process/IN/2022-04-27 Venice.mp4",
    ["ccccc", "aaaaa", "aaaaa", "aaaab", "example",
        "example", "example", "super", "keywords"],
)
