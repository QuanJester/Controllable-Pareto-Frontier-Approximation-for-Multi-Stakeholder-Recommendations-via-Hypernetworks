"""
Unit tests for checking Hypernetwork input/output shapes and gradients.
"""

import pytest
import torch
from src.models.hypernets import ParetoHypernetwork, HypernetRecommender
from src.models.backbones import MatrixFactorization, LightGCN


def test_pareto_hypernetwork_output_shape():
    batch_size = 16
    pref_dim = 3
    target_dim = 64

    hypernet = ParetoHypernetwork(
        preference_dim=pref_dim,
        hidden_dims=[32, 64],
        target_output_dim=target_dim,
    )

    dummy_pref = torch.rand(batch_size, pref_dim)
    weights = hypernet(dummy_pref)

    assert weights.shape == (batch_size, target_dim), f"Expected {(batch_size, target_dim)}, got {weights.shape}"


def test_hypernet_recommender_forward():
    num_users = 100
    num_items = 50
    emb_dim = 32
    pref_dim = 3
    batch_size = 8

    backbone = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=emb_dim)
    model = HypernetRecommender(backbone=backbone, preference_dim=pref_dim, hidden_dims=[32])

    users = torch.randint(0, num_users, (batch_size,))
    pos_items = torch.randint(0, num_items, (batch_size,))
    neg_items = torch.randint(0, num_items, (batch_size,))
    pref_vectors = torch.rand(batch_size, pref_dim)

    out = model(users, pos_items, neg_items, pref_vectors)

    assert "pos_scores" in out
    assert "neg_scores" in out
    assert "hyper_weights" in out
    assert out["pos_scores"].shape == (batch_size,)
    assert out["neg_scores"].shape == (batch_size,)
    assert out["hyper_weights"].shape == (batch_size, emb_dim)


def test_hypernet_gradients():
    pref_dim = 3
    target_dim = 16
    hypernet = ParetoHypernetwork(preference_dim=pref_dim, hidden_dims=[16], target_output_dim=target_dim)
    pref = torch.rand(4, pref_dim, requires_grad=True)

    weights = hypernet(pref)
    loss = weights.sum()
    loss.backward()

    # Kiểm tra gradient đã truyền ngược vào tham số hypernetwork
    for name, param in hypernet.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
