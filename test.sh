#!/bin/bash

# Define the list of lr values to experiment with
lr_values=(1.0e-05 5.0e-05)
a_values=(0.25 0.5 1)
b_values=(0.25 0.5 1)
d_values=(0.25 0.5 1)
train_seg_conf_th_values=(2.0 2.2 2.4)

i=0
# Loop through the lr values
for lr in "${lr_values[@]}"; do
    for a in "${a_values[@]}"; do
        for b in "${b_values[@]}"; do
            for d in "${d_values[@]}"; do
                for th in "${train_seg_conf_th_values[@]}"; do
                    ((i++))
                    result_path="/home/galinezh/TCVAE_modified/results_param_search/shang_2/shang$i"
                    echo "python3 main.py --dataset ShanghaiTech --model_optimizer adam --model_a $a --model_b $b --model_d $d --train_seg_conf_th $th --mask_root /home/galinezh/TCVAE_modified/data/ShanghaiTech/gt/test_frame_mask --seed 4423374970896992228 --vid_res 856x480 --seg_len 24 --seg_stride 12 --num_kp 18 --model_num_heads 6 --model_num_layers 6 --relative --epochs 3 --dropout 0.1 --model_lr $lr --device cuda:1 --model_save_dir $result_path --model_loss combined"
                    # Call your Python script with the current lr value
                    python3 main.py --dataset ShanghaiTech --model_optimizer adam --model_a $a --model_b $b --model_d $d --train_seg_conf_th $th --mask_root /home/galinezh/TCVAE_modified/data/ShanghaiTech/gt/test_frame_mask --seed 4423374970896992228 --vid_res 856x480 --seg_len 24 --seg_stride 12 --num_kp 18 --model_num_heads 6 --model_num_layers 6 --relative --epochs 2 --dropout 0.1 --model_lr $lr --device cuda:1 --model_save_dir $result_path --model_loss combined
                done
            done
        done
    done
done