"""PnC Stage 1 page model and Stage 2 sequence model."""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class PageClassifier(nn.Module):
    """PnC page model with Life modality adaptation."""

    def __init__(
        self,
        text_model_name,
        num_classes,
        modality="fusion",
        pretrained=True,
        clip_input_dim=1024,
        lr=1e-5,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    ):
        super().__init__()
        if modality not in {"vision", "text", "fusion"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.multimodal = modality == "fusion"
        self.embedding_dim = 1024
        self.encoder = None
        self.tokenizer = None
        if modality in {"text", "fusion"}:
            if not pretrained:
                raise ValueError("The corrected port requires pretrained PnC GBERT initialization")
            self.encoder = BertModel.from_pretrained(text_model_name, add_pooling_layer=False)
            self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        if self.multimodal:
            self.p1_encoder = nn.Linear(clip_input_dim, self.embedding_dim)
            self.dropout = nn.Dropout(0.1)
        self.out_seg = nn.Linear(self.embedding_dim, 2)
        self.out_lastpage = nn.Linear(self.embedding_dim, 2)
        self.out_class = nn.Linear(self.embedding_dim, num_classes)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def forward(
        self,
        visual_embedding=None,
        batch=None,
        return_embeddings=False,
        **kwargs,
    ):
        if batch is None:
            batch = kwargs
        elif kwargs:
            raise ValueError("Pass either batch or keyword tensors, not both")
        if self.modality == "vision":
            if visual_embedding is None:
                raise ValueError("Vision modality requires visual_embedding")
            h = visual_embedding.float()
        else:
            if self.multimodal:
                if visual_embedding is None:
                    raise ValueError("Fusion modality requires visual_embedding")
                embeddings = self.encoder.embeddings.word_embeddings(batch["input_ids"])
                embeddings[:, 0] += self.dropout(self.p1_encoder(visual_embedding.float()))
                encoded = self.encoder(
                    inputs_embeds=embeddings,
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                )
            else:
                encoded = self.encoder(**batch)
            h = encoded[0][:, 0]
        if h.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-D page representation, got {h.shape[-1]}")
        if return_embeddings:
            return h
        return self.out_seg(h), self.out_class(h), self.out_lastpage(h)

    def get_optimizer(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )


class SequenceModelRNN(nn.Module):
    def __init__(
        self,
        multimodal,
        num_classes,
        bert_input_dim=1024,
        clip_input_dim=1024,
        lr=1e-4,
        input_dim=None,
    ):
        super().__init__()
        hidden_dim = 1024
        self.input_dim = input_dim or (bert_input_dim + (clip_input_dim if multimodal else 0))
        self.multimodal = multimodal
        self.encoder = nn.GRU(
            self.input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.out_seg = nn.Linear(hidden_dim, 2)
        self.out_class = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.lr = lr

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.encoder(x)
        return self.out_seg(x), self.out_class(x)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)