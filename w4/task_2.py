import json

from track import MaxOverlapTracker
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args):
    detection_path = Path(args.detection_path)

    with open(detection_path / "coco_instances_results.json", 'r') as f:
        detection_file = json.load(f)

    tracker = MaxOverlapTracker(1, 2142)
    tracker.track_objects(detection_file)
    tracker.output_tracks(str(detection_path / "track.txt"))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a track for a detection",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "detection_path",
        type=str,
        help="Path for a test detection",
    )

    args = parser.parse_args()
    main(args)
