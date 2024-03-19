import torch, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pickle as pkl
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


DATASETS = [
    "deepsol",
]

DATASET_TO_TASK = {
    "deepsol": "single_label_classification",
}

DATASET_TO_NUM_LABELS = {
    "deepsol": 2,
}

DATASET_TO_METRICS = {
    "deepsol": [
        {
            "accuracy": Accuracy(task="multiclass", num_classes=2),
        },
        {
            "accuracy": Accuracy(task="multiclass", num_classes=2),
            "confusion": ConfusionMatrix(task="binary", num_classes=2),
        },
    ],
}

DATSET_TO_MONITOR = {
    "deepsol": "valid/accuracy",
}


class DeepSolDataset(Dataset):
    def __init__(self, seq, labels, bio_features=None):
        """
        seq:[str,..],
        struct:[str,...]
        labels: [int,..]
        bio_features: [[folat,..],...]
        """
        super().__init__()
        self.seq = seq
        self.labels = labels
        self.bio_features = bio_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        bio_feature = self.bio_features[index][:-1]
        label = int(self.labels[index])
        return self.seq[index], label, bio_feature


class WrapperData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.get_data()
        self.train_dataset = self.get_dataset(
            self.train_seq,
            self.train_labels,
            self.train_bio_features,
        )
        self.val_dataset = self.get_dataset(
            self.val_seq,
            self.val_labels,
            self.val_bio_features,
        )
        self.test_dataset = self.get_dataset(
            self.test_seq,
            self.test_labels,
            self.test_bio_features,
        )

    def get_dataset(self, seq, labels, bio_features=None):
        if self.args.dataset == "deepsol":
            return DeepSolDataset(seq, labels, bio_features)
        else:
            raise ValueError("No such dataset")

    def get_data(self):
        """
        seq_data: {label:[...], comb_seq:[...]}
        bio_features: {bio:[...]}
        """
        (
            train_data_path,
            val_data_path,
            test_data_path,
        ) = self.args.data_path
        train_df = pd.read_csv(train_data_path)
        val_df = pd.read_csv(val_data_path)
        test_df = pd.read_csv(test_data_path)
        self.train_seq = train_df["comb_seq"].values.tolist()
        self.val_seq = val_df["comb_seq"].values.tolist()
        self.test_seq = test_df["comb_seq"].values.tolist()
        self.train_labels = train_df["label"].values.tolist()
        self.val_labels = val_df["label"].values.tolist()
        self.test_labels = test_df["label"].values.tolist()

        (
            train_bio_features_path,
            val_bio_features_path,
            test_bio_features_path,
        ) = self.args.bio_feature_paths
        self.train_bio_features = pkl.load(open(train_bio_features_path, "rb"))["bio"]
        self.val_bio_features = pkl.load(open(val_bio_features_path, "rb"))["bio"]
        self.test_bio_features = pkl.load(open(test_bio_features_path, "rb"))["bio"]

    def get_trainloader(self):
        return DataLoader(
            self.train_dataset,
            self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_testloader(self):
        return DataLoader(
            self.test_dataset,
            self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_valloader(self):
        return DataLoader(
            self.val_dataset,
            self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.collator_fn,
        )

    def collator_fn(self, batch):
        seqs = []
        labels = []
        bio_features = []
        for seq, label, bio_feature in batch:
            seqs.append(seq)
            labels.append(label)
            bio_features.append(bio_feature)
        seq_batch_inputs = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.seq_max_length,
        )
        return (
            seq_batch_inputs,  # input_ids,attention_mask
            torch.tensor(labels).long(),
            torch.tensor(bio_features).float(),
        )
