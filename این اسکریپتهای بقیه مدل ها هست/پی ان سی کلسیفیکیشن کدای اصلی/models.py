import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel


class BertPageModel(nn.Module):
    """
    First stage of the pipeline: a BERT model that takes a single page as input (text + optional visual embedding)
    and predicts (1) whether it is a first page, (2) whether it is a last page, and (3) its class.
    After training, the model is used as a feature extractor (without the classification heads) for the sequence model.
    The model is trained starting from the GBERT weights.
    """

    def __init__(
        self,
        bert_model_path: str,
        multimodal: bool,
        num_classes: int,
        pretrained: bool = True,
        clip_input_dim: int = 1024,
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.multimodal = multimodal
        if pretrained:
            self.encoder = BertModel.from_pretrained(
                bert_model_path, add_pooling_layer=False
            )
        else:
            config = AutoConfig.from_pretrained(bert_model_path)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)

        if multimodal:
            self.p1_encoder = nn.Linear(clip_input_dim, 1024)
        self.out_seg = nn.Linear(1024, 2)  # First-page predictor
        self.out_lastpage = nn.Linear(1024, 2)  # Last-page predictor
        self.out_class = nn.Linear(1024, num_classes)
        if multimodal:
            self.dropout = nn.Dropout(0.1)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def forward(self, visual_embedding, batch, return_embeddings=False, **kwargs):
        if batch is None:
            assert len(kwargs) > 0
            batch = kwargs
        else:
            assert len(kwargs) == 0
        if self.multimodal:
            # Add visual embedding to [CLS] token embedding
            embs = self.encoder.embeddings.word_embeddings(batch["input_ids"])
            if self.multimodal:
                embs[:, 0] += self.dropout(self.p1_encoder(visual_embedding))
            del batch["input_ids"]
            batch["inputs_embeds"] = embs
        h = self.encoder(**batch)[0][:, 0]  # First token of last layer ([CLS])
        if return_embeddings:
            return h
        seg = self.out_seg(h)
        lp = self.out_lastpage(h)
        cl = self.out_class(h)
        return (seg, cl, lp)

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
        multimodal: bool,
        num_classes: int,
        bert_input_dim: int = 1024,
        clip_input_dim: int = 1024,
        lr: float = 0.0001,
    ):
        super().__init__()

        hidden_dim = 1024
        self.input_dim = bert_input_dim
        self.multimodal = multimodal
        if multimodal:
            self.input_dim += clip_input_dim
        self.encoder = nn.GRU(
            self.input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )

        self.out_seg = nn.Linear(hidden_dim, 2)
        self.out_class = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.lr = lr

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.encoder(x)

        x_seg = self.out_seg(x)
        x_class = self.out_class(x)
        return (x_seg, x_class)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ShuffledSampler:
    """Infinite cyclic data sampler."""

    def __init__(self, size):
        self.ptr = 0
        self.size = size
        self.shuffled_indices = None

    def _yield_batch(self, batch_size):
        for _ in range(batch_size):
            if self.shuffled_indices is None:
                self.shuffled_indices = np.random.permutation(self.size)
            next_elem = self.shuffled_indices[self.ptr]
            self.ptr += 1
            if self.ptr == self.size:
                self.ptr = 0
                self.shuffled_indices = None
            yield next_elem

    def __call__(self, batch_size):
        return list(self._yield_batch(batch_size))
