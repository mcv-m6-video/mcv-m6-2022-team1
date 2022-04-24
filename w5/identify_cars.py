import json
import pandas as pd
import wandb

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from pytorch_metric_learning import samplers, miners, losses, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CarIdDataset
from models import CarIdResnet
from utils import viz


def setup():
    parser = ArgumentParser(
        description="Create a track for a detection",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path for a configuration file",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path for the dataset",
    )
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config_path, "r") as f_json:
        cfg = json.load(f_json)
    data_path = Path(args.dataset_path)
    out_path = Path(f"./results/{cfg['exp_name']}")

    weights_path = out_path / "weights"
    logs_path = out_path / "logs"
    plots_path = out_path / "plots"

    out_path.mkdir(exist_ok=True, parents=True)
    logs_path.mkdir(exist_ok=True)
    weights_path.mkdir(exist_ok=True)
    plots_path.mkdir(exist_ok=True)

    device = torch.device(cfg["training"]["device"])

    # Setup data

    train_dataset = CarIdDataset(
        str(data_path),
        cfg["dataset"]["train_sequences"],
        "train"
    )

    aug_test_dataset = CarIdDataset(
        str(data_path),
        cfg["dataset"]["test_sequences"],
        "train"
    )
    test_dataset = CarIdDataset(
        str(data_path),
        cfg["dataset"]["test_sequences"],
        "test"
    )

    train_sampler = samplers.MPerClassSampler(
        train_dataset.get_labels(),
        batch_size=cfg["training"]["batch_size"],
        m=cfg["training"]["same_class"],
        length_before_new_iter=len(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
    )

    # Setup model
    model = CarIdResnet(cfg["model"]["embedding_layers"])
    model = model.to(device)

    # Metric learning stuff
    miner = miners.TripletMarginMiner(
        margin=cfg["training"]["mine_margin"],
        type_of_triplets=cfg["training"]["mining"]
    )
    reducer = reducers.MeanReducer(collect_stats=True)
    loss_func = losses.TripletMarginLoss(
        margin=cfg["training"]["loss_margin"],
        reducer=reducer,
    )

    acc_calculator = AccuracyCalculator(
        k='max_bin_count',
        device=device
    )

    # Optimiser
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        cfg["training"]["lr_gamma"]
    )

    # Setup WandB

    wandb.init(
        dir=str(out_path),
        entity="mthrndr",
        project="m6w5",
        config=cfg
        # config=list(pd.json_normalize(cfg).T.to_dict().values())[0]
    )

    model.train()
    for epoch in range(1, cfg["training"]["epochs"]):
        for ii, (img, labels) in enumerate(train_dataloader):
            pass
            img = img.to(device)
            features = model(img)
            tuples = miner(features, labels)
            loss = loss_func(features, labels, tuples)

            optimizer.zero_grad()
            loss.backward()
            if cfg["training"]["grad_clipping"] > 0:
                clip_grad_norm_(model.parameters(), cfg["training"]["grad_clipping"])
            optimizer.step()
            wandb.log({
                "train_loss": loss,
            })
        scheduler.step()
        if epoch % cfg["training"]["save_every"] == 0:
            torch.save(model.state_dict(), weights_path / f"weights_{epoch}.pth")
        if epoch % cfg["training"]["test_every"] == 0:
            tester = testers.BaseTester()
            model.eval()

            with torch.no_grad():
                augmented_embeddings = tester.get_all_embeddings(aug_test_dataset, model)
                reference_embeddings = tester.get_all_embeddings(test_dataset, model)

                metrics = acc_calculator.get_accuracy(
                    augmented_embeddings[0],
                    reference_embeddings[0],
                    augmented_embeddings[1].squeeze(),
                    reference_embeddings[1].squeeze(),
                    embeddings_come_from_same_source=True
                )
                viz.plot_embedding_space(
                    embeddings=reference_embeddings[0].cpu().numpy(),
                    labels=reference_embeddings[1].squeeze().cpu().numpy(),
                    epoch=epoch,
                    out_path=str(plots_path / f"embedding_{epoch}.png")
                )
            wandb.log(metrics)
            model.train()
    torch.save(model.state_dict(), weights_path / "weights_final.pth")


if __name__ == "__main__":
    main(setup())
