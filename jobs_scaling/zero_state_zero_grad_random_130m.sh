#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=zero_state_zero_grad_random_130m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18

#SBATCH --time=1-20:00:00
#SBATCH --output=./logs/updatemask%A_new_ds_0.1_1.0_zero_state_zero_grad_random_130m.out

# module purge
# module load 2023

# Your job starts in the directory where you call sbatch
cd ../
# Activate your environment
source activate test_new
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"

# Check whether the GPU is available
srun python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"
#for rank in 128 #256 512
#do

pairs=(
  "0.005 0.1"
  "0.005 0.25"
  "0.001 0.5"
  "0.001 0.75"
  "0.001 0.9"
  "0.001 1.0"
)

# Loop through each pair
for pair in "${pairs[@]}"; do
    read -r lr rank <<< "$pair"
    torchrun --standalone --nproc_per_node 1 torchrun_main_sampling1.py \
        --model_config configs/llama_130m.json \
        --lr $lr \
        --galore_scale 0.25 \
        --rank $rank \
        --update_proj_gap 100 \
        --batch_size 128  \
        --total_batch_size 512 \
        --num_training_steps 20000 \
        --warmup_steps 1000 \
        --weight_decay 0 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer galore_adamw \
        --proj_type std \
        --updating_mask_method random \
        --zero_state \
        --zero_all_state \
        --warmup_epoch 25 \
        --zero_grad  
done
#done
# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"

