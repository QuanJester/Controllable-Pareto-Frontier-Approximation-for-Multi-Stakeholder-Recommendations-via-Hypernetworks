#!/usr/bin/env bash
# ==============================================================================
# Script Grid-search khảo sát siêu tham số cho Pareto Hypernetwork
# ==============================================================================

set -e

HIDDEN_DIMS=("64,128" "128,256" "64,64,64")
LRS=(0.0005 0.001 0.002)

for dims in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LRS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Chạy Hypernetwork: hidden_dims=$dims, lr=$lr"
        echo "--------------------------------------------------------"
        python train.py \
            --config configs/model/hypernet.yaml \
            --lr "$lr" \
            --hypernet_hidden "$dims" \
            --epochs 50
    done
done

echo "Hoàn thành toàn bộ Hypernetwork Grid Search!"
