import json

from track import MaxOverlapTracker
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args):
    detection_path = Path(args.detection_path)
    nframes = args.nframes
    purge = args.purge

    out_track_name = "track.txt" if not purge else "track_purge.txt"

    with open(detection_path / "coco_instances_results.json", 'r') as f:
        detection_file = json.load(f)

    tracker = MaxOverlapTracker(1, nframes)
    tracker.track_objects(detection_file)
    if purge:
        tracker.kill_all()
        tracker.cleanup_tracks(20, 50.0)
    tracker.output_tracks(str(detection_path / out_track_name))


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
    parser.add_argument(
        "nframes",
        type=int,
        help="Number of frames in the video",
    )
    parser.add_argument(
        "--purge",
        required=False,
        action="store_true",
        default=False,
        help="perform box cleanup afterward"
    )

    args = parser.parse_args()
    main(args)
