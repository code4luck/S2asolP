from src.s2asolp import S2asolP
import pytorch_lightning as pl
import pandas as pd


class ProxyModel(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.model = S2asolP(args)
        self.predict_res = [[], []]
        self.true_labels = []

    def test(self, batch, device):
        seq_ids = batch[0]["input_ids"].to(device)
        seq_mask = batch[0]["attention_mask"].to(device)
        labels = batch[1].to(device)
        bio_features = batch[2].to(device)
        outputs = self.model(seq_ids, seq_mask, labels, bio_features)
        self.predict_res[0].extend(outputs.logits[:, 0].cpu())
        self.predict_res[1].extend(outputs.logits[:, 1].cpu())
        self.true_labels.extend(labels.cpu().numpy())
