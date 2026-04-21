#!/bin/bash

# experiment settings
DATASET=CUB200
N_CLASS=200

# hard coded inputs
GPUID='0'
CONFIG=configs/cub200_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.02
SCHEDULE=25
EMA_COEFF=0.7
SEED_LIST=(1 2 3)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10  

# FIX 1: Tạo thư mục log chuẩn xác
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

for seed in "${SEED_LIST[@]}"
do
    # save directory
    OUTDIR="./checkpoints/${DATASET}/seed${seed}"
    mkdir -p $OUTDIR

    # Create unique log file name
    LOG_FILE="${LOG_DIR}/seed${seed}.log"

    echo "Starting experiment with seed=$seed"
    
    # FIX 2: Bỏ nohup, dùng tee để in ra màn hình
    # FIX 3: Thêm --workers 0 để chống kẹt luồng
    python -u run.py \
        --config $CONFIG \
        --gpuid $GPUID \
        --repeat $REPEAT \
        --overwrite $OVERWRITE \
        --learner_type prompt \
        --learner_name APT_Learner \
        --prompt_param 0.01 \
        --lr $LR \
        --seed $seed \
        --ema_coeff $EMA_COEFF \
        --schedule $SCHEDULE \
        --workers 4 \
        --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"
        
    # Check if process completed successfully
    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully"
    else
        echo "Experiment failed"
    fi

    rm -rf ${OUTDIR}/models
    
    echo "----------------------------------------"
    
    # FIX 4: Dọn dẹp logic delay bị lỗi biến
    echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
    sleep $DELAY_BETWEEN_EXPERIMENTS
done

echo "All experiments completed!"
exit 0