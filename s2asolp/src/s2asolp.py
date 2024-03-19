import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput
from src.head_model import *


class S2asolP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.proj_dim = config.proj_dim
        self.which_head = config.pooling_head
        self.model_path = config.model_path

        self.encoder = EsmForMaskedLM.from_pretrained(self.model_path)
        if self.which_head == "mean":
            self.head = MeanHead(
                self.hidden_size,
                self.proj_dim,
                num_labels=2,
            )
        elif self.which_head == "attention1d":
            self.head = Attention1dPoolingHead(
                self.hidden_size,
                self.proj_dim,
                num_labels=2,
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, bio_features=None):
        embeddings = self.encoder.esm(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]

        if self.which_head == "mean":
            masked_features = embeddings * attention_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            embeddings = sum_features / attention_mask.sum(dim=1, keepdim=True)
            logits = self.head(embeddings, bio_features)
        elif self.which_head == "attention1d":
            logits = self.head(embeddings, attention_mask, bio_features)

        if self.training:
            loss = self.criterion(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return SequenceClassifierOutput(loss=None, logits=logits)
