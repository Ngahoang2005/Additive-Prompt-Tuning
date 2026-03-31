#!/bin/bash

# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# hard coded inputs
GPUID='0'  
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=1
OVERWRITE=0 # <-- ĐÃ ĐỔI THÀNH 1 THẬT NHÉ, ĐÉO ĐÙA NỮA!

# hyperparameter arrays
LR=0.003
SCHEDULE=30
EMA_COEFF=0.8
SEED_LIST=(1)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10

# Create log directory
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
        
        # CHẠY LIVE BẰNG TEE (Bỏ nohup và >)
        python -u run.py \
            --dataset $DATASET \
            --first_split_size 20 \
            --other_split_size 20 \
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
            --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"
        
        # Lấy exit code của lệnh python (thằng đầu tiên trong pipeline) thay vì lệnh tee
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "Experiment completed successfully"
        else
            echo "Experiment failed"
        fi

        rm -rf ${OUTDIR}/models
        
        echo "----------------------------------------"
        echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
        sleep $DELAY_BETWEEN_EXPERIMENTS
    done

echo "All experiments completed!"
exit 0