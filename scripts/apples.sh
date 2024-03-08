#! /bin/bash
count_loss="square_mape"
trials=1
ddim_steps=500
ckpt="./stable_diffusion_checkpoint/sd-v1-4.ckpt"
 
declare -a scale_list=(2)
declare -a optim_num_steps_list=(4)
declare -a optim_forward_guidance_wt_list=(400)
 
for optim_num_steps in "${optim_num_steps_list[@]}"; do
for scale in "${scale_list[@]}"; do
for optim_forward_guidance_wt in "${optim_forward_guidance_wt_list[@]}"; do
 
optim_folder=./test_object_counting/apples/50/
optim_folder+=scale_${scale}.step_${optim_num_steps}.wt_${optim_forward_guidance_wt}
 
CUDA_VISIBLE_DEVICES=0, python scripts/object_counting.py \
    --text "50 apples on the table" \
    --count_loss ${count_loss} \
    --trials ${trials} \
    --ddim_steps ${ddim_steps} \
    --optim_folder ${optim_folder} \
    --ckpt ${ckpt} \
    --scale ${scale} \
    --optim_num_steps ${optim_num_steps} \
    --optim_forward_guidance_wt ${optim_forward_guidance_wt} \
    --optim_forward_guidance \
    --optim_original_conditioning
 
done
done
done
 
