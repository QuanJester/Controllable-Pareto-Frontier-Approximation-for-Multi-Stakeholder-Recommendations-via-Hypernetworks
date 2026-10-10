"""
Module datasets: Quản lý nạp dữ liệu, tiền xử lý và sampling cho hệ thống gợi ý.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RecInteractionDataset(Dataset):
    """
    Dataset tương tác người dùng - vật phẩm (User-Item Interaction Dataset).
    Hỗ trợ negative sampling cho BPR Loss và các thông tin phụ trợ cho multi-objective.
    """

    def __init__(
        self,
        user_item_pairs: np.ndarray,
        num_users: int,
        num_items: int,
        item_features: Optional[np.ndarray] = None,
        num_negatives: int = 1,
    ):
        """
        Args:
            user_item_pairs: Mảng 2D kích thước (N, 2) chứa [user_id, pos_item_id]
            num_users: Tổng số users
            num_items: Tổng số items
            item_features: Đặc trưng bổ trợ của item (e.g. genre/category/price)
            num_negatives: Số lượng mẫu âm lấy cho mỗi mẫu dương
        """
        self.users = torch.tensor(user_item_pairs[:, 0], dtype=torch.long)
        self.pos_items = torch.tensor(user_item_pairs[:, 1], dtype=torch.long)
        self.num_users = num_users
        self.num_items = num_items
        self.item_features = (
            torch.tensor(item_features, dtype=torch.float32)
            if item_features is not None
            else None
        )
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        # Negative sampling ngẫu nhiên
        neg_item = torch.randint(0, self.num_items, (self.num_negatives,), dtype=torch.long)
        if self.num_negatives == 1:
            neg_item = neg_item.squeeze(0)

        batch = {
            "user": user,
            "pos_item": pos_item,
            "neg_item": neg_item,
        }
        return batch


def create_dataloader(
    dataset: RecInteractionDataset,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Tạo DataLoader tiện ích cho huấn luyện."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
