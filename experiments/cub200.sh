#!/bin/bash
mkdir -p /kaggle/working/data
if [ ! -d "/kaggle/working/data/CUB_200_2011" ]; then
    echo "Đang tải CUB200 dataset từ server Caltech (khoảng 1.1GB)..."
    wget -q https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz -P /kaggle/working/data/
    
    echo "Đang giải nén..."
    tar -xzf /kaggle/working/data/CUB_200_2011.tgz -C /kaggle/working/data/
    
    echo "Dọn dẹp file nén..."
    rm /kaggle/working/data/CUB_200_2011.tgz
    echo "Tải Data CUB200 xong!"
else
    echo "Data CUB200 đã tồn tại, bỏ qua bước tải."
fi
echo "----------------------------------------"

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
SEED_LIST=(2)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10  # Adjust this value as needed

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
        # Check if process completed successfully
        if [ $? -eq 0 ]; then
            echo "Experiment completed successfully"
        else
            echo "Experiment failed"
        fi

        rm -rf ${OUTDIR}/models
        
        echo "----------------------------------------"
        
        # Add delay before next experiment
        if [ $current -lt $total_experiments ]; then
            echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
            sleep $DELAY_BETWEEN_EXPERIMENTS
        fi
    done

echo "All experiments completed!"
exit 0