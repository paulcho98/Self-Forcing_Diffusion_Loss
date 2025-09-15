import argparse
import json
import torch

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, model_fn_audio


def enable_gc(model, on: bool = True, verbose: bool = True) -> bool:
    """
    Enable/disable gradient checkpointing on a loaded CausalWanModel (or PEFT-wrapped one).

    - Works whether the model is wrapped by PEFT (has `base_model`) or not.
    - Returns True if we toggled a known flag/method, else False.
    """
    try:
        base = getattr(model, 'base_model', model)
        # Preferred API when provided by model
        if hasattr(base, 'enable_gradient_checkpointing') and on:
            base.enable_gradient_checkpointing()
            if verbose:
                print("[enable_gc] Gradient checkpointing: enabled via method")
            return True
        # Try common attribute fallback
        if hasattr(base, 'gradient_checkpointing'):
            setattr(base, 'gradient_checkpointing', bool(on))
            if verbose:
                print(f"[enable_gc] Gradient checkpointing flag set to {bool(on)}")
            return True
        if verbose:
            print("[enable_gc] Model does not expose a known GC toggle")
        return False
    except Exception as e:
        if verbose:
            print(f"[enable_gc] Failed to toggle GC: {e}")
        return False


def parse_args():
    p = argparse.ArgumentParser("Verify CausalWanModel loads + (optional) LoRA + audio modules")
    p.add_argument("--model_file", type=str, required=True, help="Path to causal_model.py (defines CausalWanModel)")
    p.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pt/.pth/.safetensors)")
    p.add_argument("--adapter_weights", type=str, default=None, help="Optional trainable-only adapter (LoRA/audio) checkpoint")
    p.add_argument("--use_ema", action="store_true", help="Prefer EMA weights in checkpoint (e.g., generator_ema)")
    p.add_argument("--kwargs", type=str, default='{"use_audio": true, "in_dim": 33, "audio_hidden_size": 32}', help="JSON kwargs for CausalWanModel ctor")
    p.add_argument("--lora_rank", type=int, default=None, help="If set, apply PEFT-LoRA with this rank")
    p.add_argument("--lora_alpha", type=float, default=64.0)
    p.add_argument("--lora_targets", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--lora_init", type=str, default="kaiming")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])  # verify path only
    p.add_argument("--debug_keys", action="store_true", help="Print detailed key diffs between checkpoint and model")
    p.add_argument("--show_n", type=int, default=40, help="How many missing/unexpected keys to print in debug mode")
    p.add_argument("--check_grads", action="store_true", help="Run a tiny forward/backward to report grad norms for audio + new patch channels")
    p.add_argument("--zero_audio_proj", action="store_true", help="Zero-initialize audio_proj (ablate no-op path)")
    p.add_argument("--enable_gc", action="store_true", help="Enable gradient checkpointing on the loaded CausalWanModel")
    # Full simulation options (training-like sizes)
    p.add_argument("--simulate_training", action="store_true", help="Construct full-sized inputs (like training) and run model_fn_audio_new forward/backward")
    # Temporal sizing: either pass zipped frames directly (--frames), or RGB frames (--rgb_frames)
    p.add_argument("--frames", type=int, default=None, help="Zipped temporal length for VAE latents (Tzip). If omitted and --rgb_frames is set, computed as (rgb-1)//4+1")
    p.add_argument("--rgb_frames", type=int, default=None, help="RGB frames; used to compute zipped T=(rgb-1)//4+1 when --frames is not provided")
    p.add_argument("--frames_per_block", type=int, default=3, help="Frames per block in simulation (default 3)")
    p.add_argument("--h_lat", type=int, default=60, help="Latent height (default 60)")
    p.add_argument("--w_lat", type=int, default=104, help="Latent width (default 104)")
    p.add_argument("--no_backward", action="store_true", help="Skip backward pass in simulation")
    p.add_argument("--traceback", action="store_true", help="Print full Python traceback on failures")
    return p.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main():
    args = parse_args()
    kwargs = json.loads(args.kwargs) if args.kwargs else {}

    # Build a minimal pipeline and attach external model
    # Build a minimal pipeline without touching model registry to avoid import collisions
    pipe = WanVideoPipeline(device=args.device, torch_dtype=str_to_dtype(args.dtype))
    model = pipe.load_causal_wan(
        model_file=args.model_file,
        config_path=None,
        weights_path=args.weights,
        adapter_weights_path=args.adapter_weights,
        use_ema=args.use_ema,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets.split(',') if args.lora_targets else None,
        lora_init=args.lora_init,
        **kwargs,
    )

    # Optionally enable gradient checkpointing like training configs do
    if args.enable_gc:
        enable_gc(model, on=True, verbose=True)

    # Report basic info
    print("=== CausalWan Load Verification ===")
    print(f"Class: {model.__class__.__name__}")
    print(f"Device/dtype: {next(model.parameters()).device}/{next(model.parameters()).dtype}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M")

    # LoRA check
    has_peft = hasattr(model, 'base_model')
    print(f"PEFT-LoRA attached: {has_peft}")
    if has_peft:
        # try to find a lora module
        names = [n for n, _ in model.named_modules() if 'lora_' in n]
        print(f"LoRA modules found: {len(names)} (e.g., {names[:3]})")

    # Patch embedding shape check
    try:
        pe = model.base_model.patch_embedding if has_peft else model.patch_embedding
        w = pe.weight
        print(f"patch_embedding.weight: shape={tuple(w.shape)} mean={float(w.detach().float().mean()):.5f}")
    except Exception as e:
        print(f"patch_embedding inspect failed: {e}")

    # Audio modules check
    try:
        m = model.base_model if has_peft else model
        # Optional ablation: zero-init audio_proj
        if args.zero_audio_proj:
            try:
                apz = getattr(m, 'audio_proj', None)
                if apz is not None and hasattr(apz, 'proj') and hasattr(apz.proj, 'weight'):
                    with torch.no_grad():
                        apz.proj.weight.zero_()
                        if getattr(apz.proj, 'bias', None) is not None:
                            apz.proj.bias.zero_()
                    print("[ABLATON] audio_proj.proj zero-initialized")
            except Exception as e:
                print(f"[ABLATION] Failed to zero-init audio_proj: {e}")
        ap = getattr(m, 'audio_proj', None)
        ac = getattr(m, 'audio_cond_projs', None)
        print(f"audio_proj present: {ap is not None}")
        if ap is not None:
            proj = getattr(ap, 'proj', None)
            if proj is not None and hasattr(proj, 'weight'):
                print(f"  audio_proj.proj.weight: shape={tuple(proj.weight.shape)} mean={float(proj.weight.detach().float().mean()):.5f}")
        print(f"audio_cond_projs: {len(ac) if ac is not None else 0}")
    except Exception as e:
        print(f"audio modules inspect failed: {e}")

    print("OK: model constructed + weights loaded")

    # Optional: deep-dive into checkpoint vs model key spaces
    if args.debug_keys:
        print("\n=== Key Debug ===")
        import re, os
        from collections import Counter

        def load_ckpt_raw_state(ckpt_path: str, prefer_ema: bool = True) -> dict:
            try:
                sd = torch.load(ckpt_path, map_location='cpu')
            except TypeError:
                sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(sd, dict):
                # Self-Forcing checkpoints store under 'generator'/'generator_ema'
                key = 'generator_ema' if prefer_ema and 'generator_ema' in sd else 'generator' if 'generator' in sd else None
                if key is not None and isinstance(sd[key], dict):
                    return sd[key]
                # common alternative
                if 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                    return sd['state_dict']
            return sd if isinstance(sd, dict) else {}

        def strip_wrappers(name: str) -> str:
            return name.replace('module.', '').replace('_checkpoint_wrapped_module.', '')

        def adapt_for_peft(sd: dict, target_in_dim: int, enable_lora: bool) -> dict:
            out = {}
            for k, v in sd.items():
                nk = strip_wrappers(k)
                # Many Self-Forcing generator keys start with 'model.'
                # For PEFT-wrapped models, expect 'base_model.model.' namespace
                if enable_lora:
                    if nk.startswith('model.'):
                        nk = 'base_model.model.' + nk[len('model.'):]
                    elif nk.startswith('base_model.') and not nk.startswith('base_model.model.'):
                        nk = 'base_model.model.' + nk[len('base_model.'):]
                else:
                    # Non-PEFT: drop leading 'model.' if present
                    if nk.startswith('model.'):
                        nk = nk[len('model.'):]
                out[nk] = v

            # Remap leaf weights to '.base_layer.' for LoRA targets, if using PEFT
            if enable_lora:
                pats = [
                    re.compile(r"\.self_attn\.(q|k|v|o)\.(weight|bias)$"),
                    re.compile(r"\.cross_attn\.(q|k|v|o)\.(weight|bias)$"),
                    re.compile(r"\.ffn\.(0|2)\.(weight|bias)$"),
                ]
                mapped = {}
                for k, v in out.items():
                    mk = k
                    for pat in pats:
                        if pat.search(k) and '.base_layer.' not in k and '.lora_' not in k:
                            head, leaf = k.rsplit('.', 1)
                            mk = f"{head}.base_layer.{leaf}"
                            break
                    mapped[mk] = v
                out = mapped

            # Expand patch_embedding from 16->33 if needed
            if target_in_dim == 33:
                candidate = ('base_model.model.patch_embedding.weight' if enable_lora else 'patch_embedding.weight')
                if candidate in out:
                    w = out[candidate]
                    if hasattr(w, 'ndim') and w.ndim == 5 and w.shape[1] == 16:
                        expanded = torch.zeros(w.shape[0], 33, w.shape[2], w.shape[3], w.shape[4], dtype=w.dtype)
                        expanded[:, :16] = w
                        out[candidate] = expanded
            return out

        # Load and adapt ckpt keys
        raw = load_ckpt_raw_state(args.weights, prefer_ema=args.use_ema)
        raw_keys = set(raw.keys())
        print(f"Checkpoint raw keys: {len(raw_keys)} (sample: {list(sorted(raw_keys))[:5]})")

        target_in_dim = kwargs.get('in_dim', 16)
        enable_lora = (args.lora_rank is not None and args.lora_rank > 0)
        adapted = adapt_for_peft(raw, target_in_dim=target_in_dim, enable_lora=enable_lora)
        adapted_keys = set(adapted.keys())
        print(f"Checkpoint adapted keys: {len(adapted_keys)}")

        # Model expected keys
        model_state = model.state_dict()
        model_keys = set(model_state.keys())
        print(f"Model expected keys: {len(model_keys)}")

        # Coverage
        inter = adapted_keys & model_keys
        print(f"Key coverage: matched {len(inter)}/{len(model_keys)} ({len(inter)/max(1,len(model_keys)):.1%})")

        # Differences
        missing = sorted(list(model_keys - adapted_keys))
        unexpected = sorted(list(adapted_keys - model_keys))
        n = args.show_n
        if missing:
            print(f"Missing keys (first {min(n, len(missing))}):")
            for k in missing[:n]:
                shape = tuple(model_state[k].shape) if k in model_state else None
                print(f"  - {k} :: shape={shape}")
        if unexpected:
            print(f"Unexpected keys (first {min(n, len(unexpected))}):")
            for k in unexpected[:n]:
                v = adapted.get(k, None)
                shape = tuple(v.shape) if hasattr(v, 'shape') else None
                print(f"  - {k} :: shape={shape}")

        # Extra: spot-check shapes for a few canonical keys
        def peek(name: str):
            mshape = tuple(model_state[name].shape) if name in model_state else None
            cshape = tuple(adapted[name].shape) if name in adapted else None
            print(f"  {name}: ckpt={cshape} model={mshape}")

        print("\nShape spot-checks:")
        for nm in [
            'base_model.model.patch_embedding.weight' if enable_lora else 'patch_embedding.weight',
            'base_model.model.blocks.0.self_attn.q.base_layer.weight' if enable_lora else 'blocks.0.self_attn.q.weight',
            'base_model.model.blocks.0.ffn.0.base_layer.weight' if enable_lora else 'blocks.0.ffn.0.weight',
        ]:
            peek(nm)

        # Hint when base dims mismatch
        try:
            pe_key = 'base_model.patch_embedding.weight' if enable_lora else 'patch_embedding.weight'
            if pe_key in adapted and pe_key in model_state:
                ckpt_out = adapted[pe_key].shape[0]
                model_out = model_state[pe_key].shape[0]
                if ckpt_out != model_out:
                    print(f"\n⚠️ Detected dim mismatch in patch_embedding: ckpt_out={ckpt_out} vs model_out={model_out}.")
                    print("   Suggest setting constructor kwargs to match checkpoint architecture, e.g.:")
                    print("   --kwargs '{\"dim\": 1536, \"ffn_dim\": 8960, \"num_heads\": 12, \"num_layers\": 30, \"in_dim\": 33, \"audio_hidden_size\": 32, \"use_audio\": true}'")
        except Exception:
            pass

        # If adapter weights are provided, also show coverage for adapter file
        if args.adapter_weights:
            print("\n=== Adapter Key Debug ===")
            raw_a = load_ckpt_raw_state(args.adapter_weights, prefer_ema=False)
            adapted_a = adapt_for_peft(raw_a, target_in_dim=kwargs.get('in_dim', 16), enable_lora=enable_lora)
            keys_a = set(adapted_a.keys())
            inter_a = keys_a & model_keys
            print(f"Adapter keys: {len(keys_a)}; matched {len(inter_a)}/{len(model_keys)} ({len(inter_a)/max(1,len(model_keys)):.1%})")
            # Show some LoRA keys present in adapter
            sample_lora = [k for k in sorted(keys_a) if '.lora_' in k][:min(args.show_n, 10)]
            if sample_lora:
                print("Sample adapter LoRA keys:")
                for k in sample_lora:
                    v = adapted_a[k]
                    shape = tuple(v.shape) if hasattr(v, 'shape') else None
                    print(f"  - {k} :: shape={shape}")

    # Optional gradient check
    if args.check_grads:
        try:
            print("\n=== Gradient Check ===")
            mm = model.base_model if has_peft else model
            mm.train()
            # Ensure 33-channel input path is used
            if not hasattr(mm, 'require_vae_embedding'):
                setattr(mm, 'require_vae_embedding', True)
            else:
                mm.require_vae_embedding = True
            device = next(mm.parameters()).device
            dtype = next(mm.parameters()).dtype
            # Tiny shapes to keep it light
            B, F, H, W = 1, 4, 16, 16
            Cx, Cy = 16, 17
            Lctx, Laudio = 32, 8
            latents = torch.randn(B, Cx, F, H, W, device=device, dtype=dtype, requires_grad=False)
            y = torch.randn(B, Cy, F, H, W, device=device, dtype=dtype, requires_grad=False)
            context = torch.randn(B, Lctx, getattr(mm, 'text_dim', 4096), device=device, dtype=dtype)
            audio_emb = torch.randn(B, Laudio, 10752, device=device, dtype=dtype)
            t = torch.tensor([0.0], device=device, dtype=dtype)
            # Zero existing grads
            model.zero_grad(set_to_none=True)
            # Forward using OmniAvatar-style audio path
            out = model_fn_audio(
                dit=mm,
                latents=latents,
                timestep=t,
                context=context,
                y=y,
                audio_emb=audio_emb,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            loss = out.float().pow(2).mean()
            loss.backward()
            # Collect grad stats
            def grad_mean(tensor):
                return float(tensor.grad.detach().abs().mean().cpu()) if tensor is not None and tensor.grad is not None else 0.0
            # audio proj grad
            ap = getattr(mm, 'audio_proj', None)
            ap_w_grad = grad_mean(getattr(getattr(ap, 'proj', None), 'weight', None)) if ap is not None else 0.0
            # first audio cond proj grad
            ac = getattr(mm, 'audio_cond_projs', None)
            ac0_w_grad = grad_mean(ac[0].weight) if (ac is not None and len(ac) > 0) else 0.0
            # new patch channels grad
            pe = mm.patch_embedding
            pe_grad = 0.0
            if hasattr(pe, 'weight') and pe.weight.grad is not None:
                g = pe.weight.grad
                if g.ndim == 5 and g.shape[1] >= 33:
                    pe_grad = float(g[:, 16:, ...].detach().abs().mean().cpu())
            print(f"grad|audio_proj.proj.weight: {ap_w_grad:.6e}")
            print(f"grad|audio_cond_projs[0].weight: {ac0_w_grad:.6e}")
            print(f"grad|patch_embedding.weight[:,16:]: {pe_grad:.6e}")
            if args.zero_audio_proj:
                print("Note: zero_audio_proj enabled — expect audio path grads to be near zero.")
        except Exception as e:
            if args.traceback:
                import traceback
                traceback.print_exc()
            else:
                print(f"Gradient check failed: {e}")

    # Full training-like simulation
    if args.simulate_training:
        try:
            print("\n=== Training-like Simulation (model_fn_audio_new) ===")
            mm = model.base_model if hasattr(model, 'base_model') else model
            device = next(mm.parameters()).device
            mdtype = next(mm.parameters()).dtype
            # Resolve zipped temporal length
            if args.frames is not None:
                T = int(args.frames)
            elif args.rgb_frames is not None:
                T = (int(args.rgb_frames) - 1) // 4 + 1
                print(f"[DEBUG] computed zipped frames from rgb={args.rgb_frames}: T={T}")
            else:
                # Default to WAN Tzip for rgb=81
                T = (81 - 1) // 4 + 1
                print(f"[DEBUG] default zipped frames T={T} (from rgb=81)")
            B, H, W = 1, int(args.h_lat), int(args.w_lat)
            Cx, Cy = 16, 17
            text_len, text_dim = 512, getattr(mm, 'text_dim', 4096)
            # Ensure scheduler is in training mode with weights initialized
            try:
                need_init = False
                if not hasattr(pipe.scheduler, 'timesteps') or pipe.scheduler.timesteps is None or len(pipe.scheduler.timesteps) == 0:
                    need_init = True
                if not hasattr(pipe.scheduler, 'linear_timesteps_weights'):
                    need_init = True
                if need_init:
                    pipe.scheduler.set_timesteps(1000, training=True)
            except Exception:
                pass

            # Clean latents and noise
            clean = torch.randn(B, T, Cx, H, W, device=device, dtype=mdtype)
            noise = torch.randn_like(clean)
            # Timestep (pipeline dtype for scheduler)
            t_bf16 = pipe.scheduler.timesteps[500:501].to(dtype=pipe.torch_dtype, device=device)
            # Noisy latents and training target
            latents_sim = pipe.scheduler.add_noise(clean, noise, t_bf16)
            target_sim = pipe.scheduler.training_target(clean, noise, t_bf16)
            # y: [B, 17, T, H, W] (1-channel mask + 16-channel latents)
            mask_zip = torch.ones(B, T, H, W, device=device, dtype=pipe.torch_dtype)
            mask_zip[:, 0:1] = 0
            mask_zip = mask_zip.to(dtype=mdtype)
            y_lat = torch.randn(B, Cx, T, H, W, device=device, dtype=mdtype)
            y_sim = torch.cat([mask_zip.unsqueeze(1), y_lat], dim=1)
            # context: [B, 512, 4096]
            context_sim = torch.randn(B, text_len, text_dim, device=device, dtype=mdtype)
            # audio: expected to be RGB-aligned length before AudioPack (stride 4)
            if args.rgb_frames is not None:
                T_audio = int(args.rgb_frames)
            else:
                # Inverse mapping from zipped T to RGB ≈ (T-1)*4 + 1
                T_audio = (T - 1) * 4 + 1
            audio_sim = torch.randn(B, T_audio, 10752, device=device, dtype=mdtype)
            # Optional ablation
            if args.zero_audio_proj:
                try:
                    ap = getattr(mm, 'audio_proj', None)
                    if ap is not None and hasattr(ap, 'proj') and hasattr(ap.proj, 'weight'):
                        with torch.no_grad():
                            ap.proj.weight.zero_()
                            if getattr(ap.proj, 'bias', None) is not None:
                                ap.proj.bias.zero_()
                        print("[ABLATION] audio_proj.proj zero-initialized (simulation)")
                except Exception as e:
                    print(f"[ABLATION] Failed to zero-init audio_proj in simulation: {e}")

            # Debug: print attention/caching parameters before calling
            try:
                base = mm
                la_size = getattr(base, 'local_attn_size', -1)
                kv_size = getattr(pipe, 'kv_cache_size', None)
                fr_len = getattr(pipe, 'frame_seq_length', None)
                nblocks = getattr(pipe, 'num_transformer_blocks', None)
                print(f"[DEBUG] local_attn_size={la_size}, kv_cache_size={kv_size}, frame_seq_length={fr_len}, num_transformer_blocks={nblocks}")
                # Sanity: block size vs local window
                frames_per_block = int(args.frames_per_block)
                if la_size != -1 and la_size < frames_per_block:
                    print(f"[WARN] local_attn_size ({la_size}) < frames_per_block ({frames_per_block}); a block will not fit into KV window")
                # Sanity: total frames must not exceed KV window (for global, 21 frames → 32760 tokens)
                if la_size == -1 and isinstance(kv_size, int) and isinstance(fr_len, int):
                    max_frames = kv_size // fr_len
                    if T > max_frames:
                        print(f"[WARN] zipped frames T={T} exceeds KV window frames {max_frames}; reduce T or increase KV size")
                # Peek KV cache tensor shape by pre-initializing once
                pipe._initialize_kv_cache(B, mdtype, device)
                if isinstance(pipe.kv_cache, list) and len(pipe.kv_cache) > 0:
                    print(f"[DEBUG] kv_cache[0]['k'].shape={tuple(pipe.kv_cache[0]['k'].shape)}")
            except Exception as e:
                if args.traceback:
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"[DEBUG] cache/attn introspection failed: {e}")
            # Run forward through training-style entry
            pipe.dit = model  # ensure dit points to constructed PEFT model
            out = pipe.model_fn_audio_new(
                dit=model,
                latents=latents_sim,
                clean_latents=clean,
                timestep=t_bf16,
                context=context_sim,
                y=y_sim,
                audio_emb=audio_sim,
                use_gradient_checkpointing=args.enable_gc,
                use_gradient_checkpointing_offload=False,
            )
            print(f"output: shape={tuple(out.shape)}")
            # Compute loss and optionally backward
            loss = torch.nn.functional.mse_loss(out.float(), target_sim.float())
            try:
                w = pipe.scheduler.training_weight(t_bf16)
            except Exception:
                w = 1.0
            try:
                loss = loss * w
            except Exception:
                pass
            print(f"loss (mse * weight): {float(loss.detach().to('cpu', dtype=torch.float32)):.6f}")
            if not args.no_backward:
                model.zero_grad(set_to_none=True)
                loss.backward()
                # Report a couple grad norms
                def try_grad_mean(t):
                    return float(t.grad.detach().abs().mean().cpu()) if (t is not None and t.grad is not None) else 0.0
                ap = getattr(mm, 'audio_proj', None)
                ap_w_grad = try_grad_mean(getattr(getattr(ap, 'proj', None), 'weight', None)) if ap is not None else 0.0
                ac = getattr(mm, 'audio_cond_projs', None)
                ac0_w_grad = try_grad_mean(ac[0].weight) if (ac is not None and len(ac) > 0) else 0.0
                pe = mm.patch_embedding
                pe_new_grad = 0.0
                if hasattr(pe, 'weight') and pe.weight.grad is not None and pe.weight.grad.ndim == 5:
                    g = pe.weight.grad
                    if g.shape[1] >= 33:
                        pe_new_grad = float(g[:, 16:, ...].detach().abs().mean().cpu())
                print(f"grad|audio_proj.proj.weight: {ap_w_grad:.6e}")
                print(f"grad|audio_cond_projs[0].weight: {ac0_w_grad:.6e}")
                print(f"grad|patch_embedding.weight[:,16:]: {pe_new_grad:.6e}")
        except Exception as e:
            if args.traceback:
                import traceback
                traceback.print_exc()
            else:
                print(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
