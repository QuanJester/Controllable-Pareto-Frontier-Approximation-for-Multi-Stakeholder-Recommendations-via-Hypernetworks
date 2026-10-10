#!/usr/bin/env bash
# ==============================================================================
# Script chạy hàng loạt các phương pháp baseline (Scalarization, MGDA, Uniform)
# ==============================================================================

set -e

echo "=== [1/3] Chạy Linear Scalarization Baseline ==="
python train.py --config configs/default.yaml --method scalarization --weights 0.5 0.3 0.2

echo "=== [2/3] Chạy Multiple Gradient Descent Algorithm (MGDA) ==="
python train.py --config configs/default.yaml --method mgda

echo "=== [3/3] Chạy Equal Weight Baseline ==="
python train.py --config configs/default.yaml --method scalarization --weights 0.333 0.333 0.334

echo "Hoàn thành tất cả baseline experiments!"
