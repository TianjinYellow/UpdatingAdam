#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=test_500_5000_zero_all_state_zerograd_warmup25_re_lr_ds_75
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18

#SBATCH --time=20:00:00
#SBATCH --output=../logs_2/updatemask%A_new_500_5000_grad_max_zero_all_state_ZeroGrad_cosinedecay25_remove_lr_ds75.out

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
for i in  0.0001 0.0005 0.0008 0.001 0.005 0.008 0.01
do
    torchrun --standalone --nproc_per_node 1 torchrun_main_sampling1.py \
        --model_config configs/llama_60m.json \
        --lr $i \
        --galore_scale 0.25 \
        --rank 0.75 \
        --update_proj_gap 100 \
        --batch_size 128  \
        --total_batch_size 512 \
        --num_training_steps 10000 \
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

