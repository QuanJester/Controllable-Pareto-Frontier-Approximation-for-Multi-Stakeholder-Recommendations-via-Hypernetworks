"""
Backbone Recommender Models: Matrix Factorization, LightGCN, etc.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Mô hình Matrix Factorization (MF) cơ sở."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        scores = (u_emb * i_emb).sum(dim=-1)
        return scores


class LightGCN(nn.Module):
    """
    Mô hình LightGCN (Light Graph Convolutional Network) cho Recommendation.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Giả lập embedding propagation đơn giản hoặc tích vô hướng
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        pos_scores = (u_emb * pos_emb).sum(dim=-1)

        neg_scores = None
        if neg_items is not None:
            neg_emb = self.item_embedding(neg_items)
            neg_scores = (u_emb * neg_emb).sum(dim=-1)

        return pos_scores, neg_scores
