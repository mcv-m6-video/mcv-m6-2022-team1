import json

from track import MaxOverlapTrackerOpticalFlow
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args):
    detection_path = Path(args.detection_path)
    of_path = Path(args.opticalflow_path)
    nframes = args.nframes

    with open(detection_path / "coco_instances_results.json", 'r') as f:
        detection_file = json.load(f)

    tracker = MaxOverlapTrackerOpticalFlow(1, nframes, of_path)
    tracker.track_objects(detection_file)
    tracker.output_tracks(str(detection_path / "track_of.txt"))
    tracker.kill_all()
    tracker.cleanup_tracks(20, 50.0)
    tracker.output_tracks(str(detection_path / "track_of_purge.txt"))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a track for a detection",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "opticalflow_path",
        type=str,
        help="Path for Optical Flow extraction",
    )
    parser.add_argument(
        "detection_path",
        type=str,
        help="Path for a test detection",
    )
    parser.add_argument(
        "nframes",
        type=int,
        help="Number of frames in the video",
    )
    args = parser.parse_args()
    main(args)
