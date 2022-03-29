import json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, \
    build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances


DATA_NAME = 'ai_cities_coco'
DATA_FILE = 'gt_all'


def main(args):
    setup_logger()

    data_path = Path(args.dataset_path)
    img_path = Path(args.img_path)
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True, parents=True)
    arch_name = args.arch_name

    print(arch_name)

    register_coco_instances(
        DATA_NAME + "_test",
        {},
        str(data_path / (DATA_FILE + "_test_cocoid.json")),
        str(img_path)
    )

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(model_zoo.get_config_file(arch_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(arch_name)
    cfg.DATASETS.TEST = (DATA_NAME + "_test",)
    cfg.OUTPUT_DIR = str(out_path)
    # cfg.MODEL.WEIGHTS = str(out_path / "model_final.pth")

    """ EVALUATION """

    evaluator = COCOEvaluator(
        DATA_NAME + "_test",
        output_dir=str(out_path),
        )

    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, DATA_NAME + "_test")

    stats = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(stats)

    with open(out_path / "stats_offshelf.json", 'w') as f_stats:
        json.dump(stats, f_stats)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Fine-tune a detection network",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "img_path",
        type=str,
        help="Path to the image data",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to store output data (datasets, inference, etc)",
    )
    parser.add_argument(
        "arch_name",
        type=str,
        help="yaml file for pre-saved architecture (Detectron 2)",
    )

    args = parser.parse_args()
    main(args)
