import json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, \
    build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T

DATA_NAME = 'ai_cities'
DATA_FILE = 'gt_coco.json'


def main(args):
    setup_logger()

    data_path = Path(args.dataset_root_path)
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True, parents=True)
    arch_name = args.arch_name

    training_seqs = ["S01", "S04"]
    test_seqs = ["S03"]

    if args.train_on is not None:
        print(f"Overriden default training sequences: {args.train_on}")
        training_seqs = args.train_on
    if args.test_on is not None:
        print(f"Overriden default testing sequences: {args.test_on}")
        test_seqs = args.test_on

    train_paths = [y for x in training_seqs for y in (data_path / x).glob('*')
                   if y.is_dir()]
    test_paths = [y for x in test_seqs for y in (data_path / x).glob('*')
                  if y.is_dir()]

    train_datasets = [f"{DATA_NAME}{path.parts[-2]}{path.parts[-1]}"
                      for path in train_paths]
    test_datasets = [f"{DATA_NAME}{path.parts[-2]}{path.parts[-1]}"
                     for path in test_paths]

    print(arch_name)

    for path in train_paths + test_paths:
        register_coco_instances(
            f"{DATA_NAME}{path.parts[-2]}{path.parts[-1]}",
            {},
            str(path / "gt" / DATA_FILE),
            path / "vdo_frames",
        )

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(model_zoo.get_config_file(arch_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(arch_name)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = tuple(train_datasets)
    cfg.DATASETS.TEST = tuple(test_datasets)
    #cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = str(out_path)
    cfg.SOLVER.IMS_PER_BATCH = 4

    trainer = DefaultTrainer(cfg)
    dataloader = build_detection_train_loader(
        cfg,
        mapper=DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.RandomBrightness(0.6, 1.4),
                T.RandomFlip(prob=0.5),
                T.RandomCrop("relative_range", (0.3, 0.3)),
                T.RandomSaturation(0.5, 1.2),
                T.RandomRotation([-60, 60]),
            ])
    )

    DefaultTrainer.build_train_loader = lambda: dataloader
    trainer.resume_or_load(resume=False)
    trainer.train()

    """ EVALUATION """
    cfg.MODEL.WEIGHTS = str(out_path / "model_final.pth")

    for test_ds in test_datasets:
        print(f"Testing on {test_ds}")
        (out_path / test_ds).mkdir(exist_ok=True, parents=True)

        evaluator = COCOEvaluator(
            test_ds,
            output_dir=str(out_path / test_ds),
        )

        predictor = DefaultPredictor(cfg)
        val_loader = build_detection_test_loader(cfg, test_ds)

        stats = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(stats)

        with open(out_path / test_ds / "stats.json", 'w') as f_stats:
            json.dump(stats, f_stats)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Fine-tune a detection network",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root_path",
        type=str,
        help="Root train path for AI cities data",
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
    parser.add_argument(
        "--train_on",
        type=str,
        nargs="+",
        help="Sequences to train on. If not provided, just use default ones",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--test_on",
        type=str,
        nargs="+",
        help="Sequences to test on. If not provided, just use default ones",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    main(args)
