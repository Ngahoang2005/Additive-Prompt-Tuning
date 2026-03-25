#!/bin/bash

# 1. Cấu hình cơ bản
DATASET=CIFAR100  # Nên khớp với tên trong trainer.py
CONFIG=configs/config.yaml # Đảm bảo file này tồn tại

# 2. Sửa GPUID thành 0 (Kaggle chuẩn)
GPUID='0'
REPEAT=1
OVERWRITE=0

# 3. Tham số huấn luyện
LR=0.004
SCHEDULE=30
EMA_COEFF=0.7
SEED_LIST=(1 2 3)

# 4. Tạo thư mục log CẨN THẬN
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

for seed in "${SEED_LIST[@]}"
do
    OUTDIR="./checkpoints/${DATASET}/seed${seed}"
    mkdir -p "$OUTDIR"
    LOG_FILE="${LOG_DIR}/seed${seed}.log"

    echo "Starting experiment with seed=$seed"
    
    # Chạy trực tiếp, bỏ nohup để dễ debug trên Notebook
    python -u run.py \
        --config $CONFIG \
        --dataset $DATASET \
        --gpuid $GPUID \
        --repeat $REPEAT \
        --overwrite $OVERWRITE \
        --learner_type prompt \
        --learner_name APT_Learner \
        --prompt_param "100" "0.01" \
        --lr $LR \
        --seed $seed \
        --ema_coeff $EMA_COEFF \
        --schedule $SCHEDULE \
        --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"

    # Kiểm tra lỗi sau khi chạy xong
    if [ $? -eq 0 ]; then
        echo "Experiment with seed $seed completed successfully"
    else
        echo "Experiment with seed $seed failed. Check $LOG_FILE for details."
    fi

    echo "----------------------------------------"
    sleep 5
done

echo "All experiments completed!"