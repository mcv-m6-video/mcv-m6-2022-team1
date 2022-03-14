import xml.etree.ElementTree as ET
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args) -> None:
    video = args.video_path
    dataset = ET.parse(args.annotation_path).getroot()

    print(video, dataset)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="M6 Project: Metric test",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "annotation_path",
        type=str,
        help="Path to the dataset annotation file"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video input"
    )
    args = parser.parse_args()
    main(args)
