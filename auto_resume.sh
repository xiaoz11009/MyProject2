#!/bin/bash
# 等待当前训练进程结束后，自动续训至 80 轮
PYTHON=/home/ddd/anaconda3/envs/dgcnn/bin/python
WORKDIR=/home/ddd/zkl/Baseline2

# 等待训练完成
echo "等待训练进程结束..."
while pgrep -f "train.py" > /dev/null; do
    sleep 30
done

echo "训练完成，自动续训 30 轮 (目标 80 轮)..."
cd $WORKDIR && $PYTHON train.py \
    --batch_size 8 \
    --num_epochs 80 \
    --num_points 4096 \
    --resume ./models/best_model.pth
