import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from src.s2asolp import S2asolP


class ProxyModel(pl.LightningModule):
    def __init__(
        self,
        args,
        optim_args=None,
        metrics=(None, None),
    ):
        super().__init__()

        self.model = S2asolP(args)
        self.cf = torch.zeros((2, 2))
        self.optmi_args = optim_args
        self.valid_metrics, self.test_metrics = metrics
        self.valid_metrics = torch.nn.ModuleDict(self.valid_metrics)
        self.test_metrics = torch.nn.ModuleDict(self.test_metrics)
        self.lr = optim_args.lr
        self.file_name = args.file_name
        self.predict_res = [[], []]
        self.true_labels = []
        self.save_hyperparameters(
            ignore=[
                "tokenizer",
            ]
        )

    def training_step(self, batch, *args, **kwargs):
        seq_ids = batch[0]["input_ids"]
        seq_mask = batch[0]["attention_mask"]

        labels = batch[1]
        bio_features = batch[2]
        outputs = self.model(seq_ids, seq_mask, labels, bio_features)

        self.log(
            "train/loss",
            outputs.loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train/lr", lr, logger=True, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "train/step",
            self.global_step,
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return outputs.loss

    def validation_step(self, batch, *args, **kwargs):
        seq_ids = batch[0]["input_ids"]
        seq_mask = batch[0]["attention_mask"]
        labels = batch[1]
        bio_features = batch[2]
        outputs = self.model(seq_ids, seq_mask, labels, bio_features)
        for name, metric in self.valid_metrics.items():
            self.log(
                f"valid/{name}",
                metric(outputs.logits, labels.long()),
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def on_test_epoch_end(self) -> None:
        df = pd.DataFrame(
            {
                "true_labels": self.true_labels,
                "pred_insolubility": self.predict_res[0],
                "pred_solubility": self.predict_res[1],
            }
        )
        df.to_csv("./{}_predict_res.csv".format(self.file_name), index=False)

    def test_step(self, batch, *args, **kwargs):
        seq_ids = batch[0]["input_ids"]
        seq_mask = batch[0]["attention_mask"]

        labels = batch[1]
        bio_features = batch[2]
        outputs = self.model(seq_ids, seq_mask, labels, bio_features)

        self.predict_res[0].extend(outputs.logits[:, 0].cpu().numpy())
        self.predict_res[1].extend(outputs.logits[:, 1].cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())
        for name, metric in self.test_metrics.items():
            if name == "confusion":
                logits = torch.argmax(outputs.logits, dim=-1)
                self.cf += metric(logits, labels.long()).to(self.cf.device)
            elif name == "accuracy":
                self.log(
                    f"test/{name}",
                    metric(outputs.logits, labels.long()),
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        if self.optmi_args.finetune == "head":
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            pass
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.optmi_args.weight_decay,
        )
        lr_sch = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=0.998, step_size=5, last_epoch=-1
        )
        return optimizer
