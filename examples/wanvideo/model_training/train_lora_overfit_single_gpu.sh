#!/bin/bash

# Minimal overfit run on 1–3 clips to sanity-check training.
# Uses the same train.py but points to a tiny metadata file.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port 29501 "$THIS_DIR/train.py" \
  --dataset_base_path="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/videos_cfr" \
  --dataset_metadata_path="$THIS_DIR/metadata_audio_micro.csv" \
  --data_file_keys="video" \
  --height=480 \
  --width=832 \
  --num_frames=81 \
  --trainable_models="dit" \
  --model_paths='["/home/cvlab20/project/jinhyuk/DiffSynth-Studio/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors","/home/cvlab20/project/hyunbin/talkingface_dmd/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/home/cvlab20/project/hyunbin/talkingface_dmd/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --lora_base_model="dit" \
  --lora_target_modules="q,k,v,o,ffn.0,ffn.2" \
  --lora_rank=128 \
  --remove_prefix_in_ckpt="pipe.dit." \
  --output_path="/mnt/dataset1/hyunbin/self_forcing_omniavatar_diffusion_loss/overfit_1clip" \
  --learning_rate=5e-5 \
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
  --use_wandb \
  --wandb_project="Self-Forcing-OmniAvatar-Diffusion-Loss" \
  --wandb_entity="paulhcho" \
  --wandb_log_every 4\
  --wandb_run_name "overfit_1clip_audio"\
  --use_causal_wan \
  --causal_wan_model_file "/home/cvlab20/project/hyunbin/talkingface_dmd/Self-Forcing/wan/modules/causal_model.py" \
  --causal_wan_weights "/home/cvlab20/project/hyunbin/talkingface_dmd/Self-Forcing/checkpoints/self_forcing_dmd.pt" \
  --causal_wan_lora_rank 128 \
  --causal_wan_lora_alpha 64 \
  --causal_wan_lora_targets "q,k,v,o,ffn.0,ffn.2" \
  --causal_wan_kwargs '{"use_audio": true, "in_dim": 33, "audio_hidden_size": 32}'\
  --enable_gc \

echo "Overfit run started. Check ./output/overfit_1clip for checkpoints."
