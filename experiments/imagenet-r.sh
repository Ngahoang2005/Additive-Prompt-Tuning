#!/bin/bash

# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# hard coded inputs
GPUID='0'  # <-- SỬA: Đổi về 0 vì Kaggle chỉ có 1 GPU
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.003
SCHEDULE=30
EMA_COEFF=0.8
SEED_LIST=(1)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10

# Create log directory
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"  # <-- SỬA: Tạo thư mục log đầy đủ đường dẫn

for seed in "${SEED_LIST[@]}"
    do
        # save directory
        OUTDIR="./checkpoints/${DATASET}/seed${seed}"
        mkdir -p $OUTDIR

        # Create unique log file name
        LOG_FILE="${LOG_DIR}/seed${seed}.log"

        echo "Starting experiment with seed=$seed"
        
        nohup python -u run.py \
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
            --dataroot /kaggle/working/data \
            --log_dir ${OUTDIR} > "$LOG_FILE" 2>&1 &

        # Store the PID of the background process
        PID=$!
        
        # Wait for process to complete
        wait $PID
        
        # Check if process completed successfully
        if [ $? -eq 0 ]; then
            echo "Experiment completed successfully"
        else
            echo "Experiment failed"
        fi

        rm -rf ${OUTDIR}/models
        
        echo "----------------------------------------"
        
        # <-- SỬA: Xóa cái if lỗi biến rỗng đi, cho sleep luôn
        echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
        sleep $DELAY_BETWEEN_EXPERIMENTS
    done

echo "All experiments completed!"
exit 0