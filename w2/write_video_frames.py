import ffmpeg
import numpy
import cv2
import sys
import random
from pathlib import Path


def read_frame_as_jpeg(in_file, frame_num):
    """
    Read any frame with specified number of frames
    """
    out, err = (
        ffmpeg.input(in_file)
              .filter('select', 'gte(n,{})'.format(frame_num))
              .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
              .run(capture_stdout=True)
    )
    return out


def get_video_info(in_file):
    """
    Get basic video information
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


file_path = Path('E:/Master/M6 - Video analysis/Project/AICity_data/train/S03/c010/vdo.avi')
video_info = get_video_info(file_path)
total_frames = int(video_info['nb_frames'])
print('total frames: '+ str(total_frames))
random_frame = random.randint(1, total_frames)
print('random frame: '+ str(random_frame))
out = read_frame_as_jpeg(file_path, random_frame)
image_array = numpy.asarray(bytearray(out), dtype="uint8")
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
cv2.imshow('frame', image)
cv2.waitKey()

# ffmpeg -i E:/Master/M6 - Video analysis/Project/AICity_data/train/S03/c010/vdo.avi -r 1/1 $filename%03d.jpg
