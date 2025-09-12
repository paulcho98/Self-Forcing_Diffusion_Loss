#!/bin/bash

# Minimal overfit run on 1–3 clips to sanity-check training.
# Uses the same train.py but points to a tiny metadata file.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

accelerate launch "$THIS_DIR/train.py" \
  --dataset_base_path="/home/work/.local/cropped_only_10K_preprocessed" \
  --dataset_metadata_path="$THIS_DIR/metadata_audio_micro.csv" \
  --data_file_keys="video" \
  --height=480 \
  --width=832 \
  --num_frames=81 \
  --trainable_models="dit" \
  --model_paths='["/home/work/.local/Diffusion-Loss/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors","/home/work/.local/Self-Forcing-Omniavatar/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/home/work/.local/Self-Forcing-Omniavatar/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --lora_base_model="dit" \
  --lora_target_modules="q,k,v,o,ffn.0,ffn.2" \
  --lora_rank=128 \
  --remove_prefix_in_ckpt="pipe.dit." \
  --output_path="./output/overfit_1clip" \
  --learning_rate=1e-4 \
  --num_epochs=20 \
  --dataset_repeat=200 \
  --save_steps=50 \
  --extra_inputs="input_image,audio_emb" \
  --gradient_accumulation_steps=1 \
  --sf_restrict_timesteps \
  --sf_denoising_step_list="1000,750,500,250" \
  --sf_warp_denoising_step \
  --sf_timestep_shift=5.0 \
  --dataset_num_workers=0 \
  --find_unused_parameters

echo "Overfit run started. Check ./output/overfit_1clip for checkpoints."
