import argparse
import os
import torch

# Reuse training module and dataset to stay consistent with training loop
from train import WanTrainingModule
from diffsynth.trainers.unified_dataset import UnifiedDataset


def parse_args():
    p = argparse.ArgumentParser("Verify that audio embeddings affect gradients in training path")
    # Dataset
    p.add_argument("--dataset_base_path", type=str, default="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/videos_cfr")
    p.add_argument("--dataset_metadata_path", type=str, default="metadata_audio.csv")
    # Match training: do NOT include prompt/audio_emb so they are not processed as files
    p.add_argument("--data_file_keys", type=str, default="image,video")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--max_pixels", type=int, default=1920 * 1080)
    p.add_argument("--dataset_repeat", type=int, default=1)

    # Model
    p.add_argument(
        "--model_paths",
        type=str,
        default='["/home/cvlab20/project/jinhyuk/DiffSynth-Studio/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors","/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]'
    )
    p.add_argument("--trainable_models", type=str, default="dit")
    p.add_argument("--lora_base_model", type=str, default="dit")
    p.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit")
    p.add_argument("--extra_inputs", type=str, default="input_image,audio_emb")

    # Device
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])  # used for inputs
    p.add_argument("--sample_index", type=int, default=0)
    # Determinism and timestep control
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fix_timestep", type=int, default=500, help="Fix the training timestep index (0-999). Use None to disable.")
    # Optionally break zero-init deadlock by warming audio_proj only
    p.add_argument("--warm_audio_proj", action="store_true", help="Initialize audio_proj.proj.weight with small normal noise.")
    p.add_argument("--warm_std", type=float, default=1e-3, help="Std-dev for audio_proj warm-up init.")
    return p.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main():
    args = parse_args()

    # Seed for determinism
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build dataset similar to training
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    data = dataset[args.sample_index % len(dataset)]
    if "audio_emb" not in data or data["audio_emb"] is None:
        print("[ERROR] The selected sample has no 'audio_emb' entry in metadata. Pick a different row.")
        return

    # Build model like training, then move to desired device
    model = WanTrainingModule(
        model_paths=args.model_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        extra_inputs=args.extra_inputs,
        dataset_base_path=args.dataset_base_path,
        # keep SF settings default; they don’t matter here
    )

    # Move modules to device and dtype similar to training runtime
    device = torch.device(args.device)
    target_dtype = str_to_dtype(args.dtype)
    # Ensure all submodules (including newly created Conv3d/Linear) match dtype/device
    model.pipe.to(device=device, dtype=target_dtype)
    model.pipe.device = device.type
    # Keep pipeline compute dtype in sync
    model.pipe.torch_dtype = target_dtype

    # Fix timestep so both passes compare at the same denoising step
    if args.fix_timestep is not None:
        ts = int(args.fix_timestep)
        if ts < 0 or ts > 999:
            raise ValueError("--fix_timestep must be in [0, 999]")
        model.pipe.sf_allowed_timestep_indices = torch.tensor([ts], dtype=torch.long)

    # Optionally warm audio_proj to avoid zero-grad deadlock while keeping cond heads at zero
    if args.warm_audio_proj:
        dit = model.pipe.dit
        if hasattr(dit, "audio_proj") and hasattr(dit.audio_proj, "proj"):
            with torch.no_grad():
                torch.nn.init.normal_(dit.audio_proj.proj.weight, mean=0.0, std=args.warm_std)
                if getattr(dit.audio_proj.proj, "bias", None) is not None:
                    dit.audio_proj.proj.bias.zero_()

    # Prepare inputs (this loads audio tensor from path and pads to num_frames)
    inputs = model.forward_preprocess(data)

    # Cast input tensors to chosen dtype/device to avoid implicit casts
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device=device, dtype=target_dtype)

    # Ensure audio is present and get shape
    audio = inputs.get("audio_emb", None)
    if audio is None:
        print("[ERROR] forward_preprocess did not produce 'audio_emb'. Ensure --extra_inputs includes audio_emb and metadata has that column.")
        return

    # Utility to get grad norms of audio modules
    def audio_grad_stats(dit):
        stats = {}
        if hasattr(dit, "audio_proj") and hasattr(dit.audio_proj, "proj"):
            w = dit.audio_proj.proj.weight
            stats["audio_proj.proj.weight"] = float(w.grad.norm().item()) if w.grad is not None else 0.0
            stats["audio_proj.proj.weight_abs_mean"] = float(w.detach().abs().mean().item())
        if hasattr(dit, "audio_cond_projs") and dit.audio_cond_projs is not None:
            vals = []
            abs_means = []
            for i, lin in enumerate(dit.audio_cond_projs):
                g = lin.weight.grad
                vals.append(float(g.norm().item()) if g is not None else 0.0)
                abs_means.append(float(lin.weight.detach().abs().mean().item()))
            stats["audio_cond_projs[*].weight"] = vals
            stats["audio_cond_projs[*].weight_abs_mean"] = abs_means
        return stats

    # Compute loss and grads with real audio
    model.train()
    model.zero_grad(set_to_none=True)
    loss_real = model.forward(data=None, inputs=inputs)
    loss_real.backward()
    stats_real = audio_grad_stats(model.pipe.dit)

    # Compute loss and grads when audio is zeroed
    model.zero_grad(set_to_none=True)
    inputs_zero = dict(inputs)
    inputs_zero["audio_emb"] = torch.zeros_like(audio)
    loss_zero = model.forward(data=None, inputs=inputs_zero)
    loss_zero.backward()
    stats_zero = audio_grad_stats(model.pipe.dit)

    # Report
    print("=== Audio Flow Verification ===")
    print(f"Sample index: {args.sample_index}")
    print(f"Audio shape: {tuple(audio.shape)} (B, L, D)")
    print(f"Loss with real audio: {float(loss_real.item()):.6f}")
    print(f"Loss with zero audio: {float(loss_zero.item()):.6f}")
    print("Grad norms (real audio):", stats_real)
    print("Grad norms (zero audio):", stats_zero)

    # Simple heuristic checks
    any_grad_real = (stats_real.get("audio_proj.proj.weight", 0.0) > 0.0) or any(x > 0.0 for x in stats_real.get("audio_cond_projs[*].weight", []))
    any_grad_zero = (stats_zero.get("audio_proj.proj.weight", 0.0) > 0.0) or any(x > 0.0 for x in stats_zero.get("audio_cond_projs[*].weight", []))
    print("Pass condition: grads(nonzero with real audio) AND grads(zero or much smaller with zero audio)")
    print(f"Any audio grads with real audio? {any_grad_real}")
    print(f"Any audio grads with zero audio? {any_grad_zero}")

    # Optional: Difference in loss provides additional signal (may be equal if audio layers are zero-initialized)
    print(f"Loss delta (real - zero): {float((loss_real - loss_zero).abs().item()):.6f}")


if __name__ == "__main__":
    main()
