#!/bin/bash

# This script fine-tunes a local Wan 2.1 1.3B model
# with LoRA on your custom dataset.

accelerate launch ./train.py \
    --dataset_base_path="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/videos_cfr" \
    --dataset_metadata_path="metadata_audio.csv" \
    --height=480 \
    --width=832 \
    --num_frames=81 \
    --trainable_models="dit" \
    --model_paths='["/home/cvlab20/project/jinhyuk/DiffSynth-Studio/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors","/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
    --lora_base_model="dit" \
    --lora_target_modules="q,k,v,o,ffn.0,ffn.2" \
    --lora_rank=128 \
    --remove_prefix_in_ckpt="pipe.dit" \
    --output_path="./output/self_forcing_omniavatar_diffusion_loss" \
    --learning_rate=5e-5 \
    --num_epochs=10 \
    --save_steps=100 \
    --extra_inputs="input_image,audio_emb" \
    --gradient_accumulation_steps=32 \
    --sf_restrict_timesteps \
    --sf_denoising_step_list="1000,750,500,250" \
    --sf_warp_denoising_step \
    --sf_timestep_shift=5.0 \
    --use_wandb \
    --wandb_project="Self-Forcing-OmniAvatar-Diffusion-Loss" \
    --wandb_entity="paulhcho" \
    --wandb_log_every 32\
    # --use_gradient_checkpointing_offload \
    \
    # # --- MEMORY SAVING CHANGES ---
    # # 1. Reduce the per-device batch size to the minimum.
    # --per_device_train_batch_size=1 \
    # \
    # # 2. Use gradient accumulation to maintain a larger effective batch size.
    # #    An effective batch size of 4 (1 * 4) is a good starting point.
    # --gradient_accumulation_steps=4