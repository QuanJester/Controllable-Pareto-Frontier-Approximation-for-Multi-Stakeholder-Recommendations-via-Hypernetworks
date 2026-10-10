# Pareto Approximation Research

Nghiên cứu xấp xỉ mặt tiền Pareto (Pareto Frontier Approximation) trong Hệ thống Gợi ý Đa mục tiêu (Multi-Objective Recommender Systems - MSR) sử dụng Hypernetworks và các phương pháp Tối ưu hóa Đa mục tiêu (Multi-Objective Optimization - MOO).

---

## 📁 Cấu trúc Dự án (Project Structure)

```text
Pareto Approximation Research/
├── configs/                  # File cấu hình (YAML/JSON/Hydra)
│   ├── model/               # Cấu hình kiến trúc: hypernet.yaml, base_rec.yaml
│   ├── dataset/             # Cấu hình dataset: movielens.yaml, amazon.yaml
│   ├── experiment/          # Gom config cho từng thí nghiệm cụ thể
│   └── default.yaml         # Config mặc định chung
│
├── data/                     # Dữ liệu (tuyệt đối add vào .gitignore)
│   ├── raw/                 # Dữ liệu gốc tải về, read-only
│   ├── processed/           # Dữ liệu sau khi xử lý (split train/val/test)
│   └── synthetic/           # Dữ liệu test/toy chạy thử nghiệm nhanh
│
├── src/                      # Mã nguồn chính của dự án (đóng gói dạng package)
│   ├── __init__.py
│   ├── datasets/            # Data loader, sampler, tiền xử lý
│   ├── models/              # Định nghĩa kiến trúc PyTorch
│   │   ├── backbones/       # Mô hình gợi ý gốc (MF, LightGCN, SASRec,...)
│   │   ├── hypernets/       # Mạng Hypernetwork sinh trọng số
│   │   └── baselines/       # Các phương pháp so sánh (Scalarization, MGDA, EPO,...)
│   ├── objectives/          # Hàm loss, Pareto metrics (Hypervolume, Spread, Spacing)
│   ├── utils/               # Seed, logger, visualization helpers
│   └── trainer.py           # Vòng lặp huấn luyện, validation, early stopping
│
├── experiments/              # Quản lý chạy thực nghiệm
│   ├── scripts/             # Bash scripts (.sh) chạy hàng loạt hoặc chạy SLURM/cluster
│   │   ├── run_baselines.sh
│   │   └── run_hypernet_grid.sh
│   └── notebooks/           # Jupyter Notebooks (CHỈ dùng để EDA hoặc vẽ chart báo cáo)
│       ├── 01_eda_stakeholder_bias.ipynb
│       └── 02_pareto_frontier_plots.ipynb
│
├── outputs/                  # Kết quả xuất ra (add vào .gitignore)
│   ├── checkpoints/         # Model weights (.pt, .ckpt) lưu theo run_id/timestamp
│   ├── logs/                # TensorBoard logs hoặc log file text
│   └── figures/             # Vector đồ họa (PDF/SVG) xuất ra để đưa vào paper
│
├── paper/                    # Tài liệu và bài báo (quản lý bằng Git riêng hoặc Overleaf)
│   ├── figures/             # Đồ họa dùng trực tiếp trong LaTeX
│   ├── sections/            # Tách nhỏ file .tex (intro.tex, method.tex,...)
│   ├── references.bib       # Trích dẫn khoa học
│   └── main.tex
│
├── tests/                    # Unit tests kiểm tra shape, gradient, metrics
│   ├── test_hypernet_shapes.py
│   └── test_pareto_metrics.py
│
├── .gitignore
├── README.md                 # Hướng dẫn setup môi trường và reproduce kết quả
├── requirements.txt          # Hoặc environment.yml / pyproject.toml
└── train.py                  # Entrypoint chính để chạy huấn luyện
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Setup & Usage)

### 1. Cài đặt Môi trường (Environment Setup)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
# source venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình (Training)
```bash
# Huấn luyện mô hình Hypernetwork với config mặc định
python train.py --config configs/default.yaml

# Huấn luyện mô hình cụ thể (chọn model & dataset)
python train.py --model hypernet --dataset movielens --epochs 50 --lr 0.001
```

### 3. Chạy Thực nghiệm Hàng loạt (Batch Experiments)
```bash
# Chạy các baseline so sánh
bash experiments/scripts/run_baselines.sh

# Chạy grid-search khảo sát Hypernetwork
bash experiments/scripts/run_hypernet_grid.sh
```

### 4. Kiểm thử Đơn vị (Unit Tests)
```bash
pytest tests/ -v
```

---

## 🎯 Mục tiêu Nghiên cứu
- **Đa mục tiêu (Multi-Objective Recommender Systems)**: Cân bằng giữa Accuracy (Độ chính xác), Diversity (Độ đa dạng), Novelty (Độ mới mẻ), và Fairness (Công bằng giữa các bên liên quan - Stakeholders).
- **Hypernetworks**: Học ánh xạ liên tục từ vector sở thích (preference vector) $r \in \Delta^m$ sang không gian tham số của mô hình gợi ý, cho phép sinh toàn bộ đường biên Pareto chỉ với 1 mô hình duy nhất khi inference.
- **Pareto Metrics**: Đánh giá chất lượng tập xấp xỉ bằng Hypervolume (HV), Spacing (SP), Spread $(\Delta)$.
