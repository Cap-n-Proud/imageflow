# from scenedetect import detect, ContentDetector, AdaptiveDetector
#
# scene_list = detect("in.MOV", AdaptiveDetector(threshold=1))
# for i, scene in enumerate(scene_list):
#     print(
#         "    Scene %2d: Start %s / Frame %d, End %s / Frame %d"
#         % (
#             i + 1,
#             scene[0].get_timecode(),
#             scene[0].get_frames(),
#             scene[1].get_timecode(),
#             scene[1].get_frames(),
#         )
#     )

from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images


def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
    # scene_manager.add_detector(AdaptiveDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    save_images(scene_list=scene_list, video=video, output_dir="./", num_images=1)


split_video_into_scenes("./in.MOV", 1)
