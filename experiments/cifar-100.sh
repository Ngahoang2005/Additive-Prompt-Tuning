#!/bin/bash

# experiment settings
DATASET=cifar-100
N_CLASS=100

# hard coded inputs
GPUID='0'
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.004
SCHEDULE=30
EMA_COEFF=0.7
SEED_LIST=(1 2 3)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10  # Adjust this value as needed

# Create log directory
LOG_DIR="logs"
mkdir -p ${LOG_DIR}/${DATASET}
current=1
total_experiments=${#SEED_LIST[@]}
for seed in "${SEED_LIST[@]}"
    do
        # save directory
        OUTDIR="./checkpoints/${DATASET}/seed${seed}"
        mkdir -p $OUTDIR

        # Create unique log file name
        LOG_FILE="${LOG_DIR}/${DATASET}/seed${seed}.log"

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
            --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"   

        # Store the PID of the background process
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Experiment completed successfully"
    else
        echo "Experiment failed"
        exit 1 # Lỗi thì dừng luôn cho an toàn
    fi

    rm -rf ${OUTDIR}/models
    
    echo "----------------------------------------"
    
    # Add delay before next experiment
    if [ $current -lt $total_experiments ]; then
        echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
        sleep $DELAY_BETWEEN_EXPERIMENTS
    fi
    current=$((current+1))
done

echo "All experiments completed!"
exit 0