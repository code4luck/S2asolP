from pytorch_lightning import Trainer
from argparse import ArgumentParser
from src.wrapperdataset import *
import torch
from proxy_model import ProxyModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
from src.utils import get_diff_args

warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="deepsol", choices=DATASETS)
    parser.add_argument("--file_name", type=str, default="s2solp")
    parser.add_argument("--data_path", nargs="+", type=str, default=None)
    parser.add_argument("--bio_feature_paths", nargs="+", type=str, default=None)
    parser.add_argument("--split_method", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="saprot_pdb")
    parser.add_argument("--tokenizer", type=str, default="saprot_pdb")
    # model
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--pooling_head", type=str, default="mean")
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--proj_dim", type=int, default=1280)
    parser.add_argument("--is_drop", type=bool, default=True)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--seq_max_length", type=int, default=1200)
    # wandb
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # optimizer
    parser.add_argument("--optim_lr", type=float, default=1e-4)
    parser.add_argument("--optim_weight_decay", type=float, default=0.001)
    parser.add_argument(
        "--optim_finetune", type=str, default="all", choices=["all", "head", "lora"]
    )

    # Trainer
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--precision", type=str, default="32")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_path
    return args


def main():
    from pytorch_lightning.utilities import rank_zero_info

    torch.set_float32_matmul_precision("high")
    args = parse_args()
    wrapper_dataset = WrapperData(args)
    train_loader = wrapper_dataset.get_trainloader()
    valid_loader = wrapper_dataset.get_valloader()
    test_loader = wrapper_dataset.get_testloader()

    rank_zero_info(f"Dataset: {args.dataset}")
    rank_zero_info(f"Train: {len(wrapper_dataset.train_dataset)}")
    rank_zero_info(f"Valid: {len(wrapper_dataset.val_dataset)}")
    rank_zero_info(f"Test: {len(wrapper_dataset.test_dataset)}")

    optim_args = get_diff_args(args, mode="optim_")
    print("optim_args ", optim_args)
    model = ProxyModel(
        args=args,
        optim_args=optim_args,
        metrics=DATASET_TO_METRICS[args.dataset],
    )

    monitor = DATSET_TO_MONITOR[args.dataset]
    m = "s2asolp"
    ckpt_name = (
        f"{args.dataset}-{args.split_method}-{m}-{args.pooling_head}"
        if args.split_method
        else f"{args.dataset}-{m}-{args.pooling_head}"
    )
    if args.wandb_run_name is None:
        args.wandb_run_name = ckpt_name
    mode = "max"
    model_checkpoint = ModelCheckpoint(
        "checkpoints/s2asolp",
        monitor=DATSET_TO_MONITOR[args.dataset],
        mode=mode,
        filename=ckpt_name,
        verbose=True,
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        logger=(
            [
                WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config=vars(args),
                )
            ]
            if args.wandb
            else None
        ),
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        deterministic=True,
        precision=args.precision,
        callbacks=[
            model_checkpoint,
            EarlyStopping(
                monitor=monitor, mode="max", patience=args.patience, verbose=True
            ),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
    model = ProxyModel.load_from_checkpoint(model_checkpoint.best_model_path)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
