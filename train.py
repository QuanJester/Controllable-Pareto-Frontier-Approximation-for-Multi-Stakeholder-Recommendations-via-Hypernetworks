"""
Entrypoint chính để chạy huấn luyện mô hình Pareto Approximation trong Recommender Systems.
Usage:
    python train.py --config configs/default.yaml
    python train.py --model hypernet --dataset movielens --epochs 20 --lr 0.001
"""

import os
import argparse
import yaml
import numpy as np
import torch

from src.utils import set_seed, setup_logger
from src.datasets import RecInteractionDataset, create_dataloader
from src.models import LightGCN, HypernetRecommender
from src.trainer import ParetoTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Pareto Approximation Recommender")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn tới file cấu hình yaml")
    parser.add_argument("--model", type=str, default="hypernet", choices=["hypernet", "base_rec"], help="Loại mô hình")
    parser.add_argument("--dataset", type=str, default="movielens", help="Tên tập dữ liệu")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=None, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=None, help="Tốc độ học (learning rate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: 'cuda' hoặc 'cpu'")
    parser.add_argument("--hypernet_hidden", type=str, default="64,128", help="Kích thước hidden layers cho Hypernet (phân tách bởi dấu phẩy)")
    parser.add_argument("--method", type=str, default="hypernet", help="Phương pháp tối ưu: 'hypernet', 'scalarization', 'mgda'")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Trọng số cho Scalarization")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    logger = setup_logger("Main")
    
    # Nạp cấu hình từ file YAML
    cfg = load_config(args.config)
    seed = args.seed or cfg.get("seed", 42)
    set_seed(seed)
    
    device_name = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng, tự động chuyển sang CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(f"Sử dụng thiết bị: {device}")

    training_cfg = cfg.get("training", {})
    epochs = args.epochs or training_cfg.get("epochs", 20)
    batch_size = args.batch_size or training_cfg.get("batch_size", 512)
    lr = args.lr or training_cfg.get("learning_rate", 0.001)

    logger.info(f"Cấu hình chạy: Model={args.model}, Epochs={epochs}, BatchSize={batch_size}, LR={lr}")

    # Giả lập dữ liệu đồ chơi (Toy/Synthetic Dataset) để kiểm tra luồng huấn luyện nhanh
    num_users = 200
    num_items = 100
    num_interactions = 2000
    np.random.seed(seed)
    dummy_users = np.random.randint(0, num_users, size=(num_interactions, 1))
    dummy_items = np.random.randint(0, num_items, size=(num_interactions, 1))
    dummy_pairs = np.hstack([dummy_users, dummy_items])

    train_data = RecInteractionDataset(dummy_pairs[:1600], num_users=num_users, num_items=num_items)
    val_data = RecInteractionDataset(dummy_pairs[1600:], num_users=num_users, num_items=num_items)

    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình
    backbone = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=64)

    if args.model == "hypernet":
        hidden_dims = [int(x) for x in args.hypernet_hidden.split(",")]
        model = HypernetRecommender(
            backbone=backbone,
            preference_dim=3,
            hidden_dims=hidden_dims,
        )
        logger.info(f"Đã khởi tạo Pareto Hypernetwork với hidden_dims={hidden_dims}")
    else:
        model = backbone
        logger.info("Đã khởi tạo Backbone Recommender tiêu chuẩn.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = ParetoTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_objectives=3,
        device=str(device),
        checkpoint_dir=cfg.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints"),
        patience=training_cfg.get("early_stopping_patience", 5),
        logger=logger,
    )

    trainer.fit(epochs=epochs)
    logger.info("Huấn luyện hoàn tất!")


if __name__ == "__main__":
    main()
