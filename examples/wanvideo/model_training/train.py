import torch, os, json
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.audio_pack import AudioPack
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable lightweight memory debugging via env var MEM_DEBUG=1
MEM_DEBUG = os.environ.get("MEM_DEBUG", "0") == "1"

def _bytes_to_gb(x: int | float) -> float:
    try:
        return float(x) / (1024 ** 3)
    except Exception:
        return float(x)

def _gpu_mem_report(tag: str = "", device: torch.device | str = "cuda"):
    if not MEM_DEBUG or not torch.cuda.is_available():
        return
    try:
        dev = torch.device(device)
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        max_alloc = torch.cuda.max_memory_allocated(dev)
        max_reserv = torch.cuda.max_memory_reserved(dev)
        print(f"[MemDbg][{tag}] allocated={_bytes_to_gb(alloc):.2f}GB reserved={_bytes_to_gb(reserv):.2f}GB max_alloc={_bytes_to_gb(max_alloc):.2f}GB max_reserved={_bytes_to_gb(max_reserv):.2f}GB")
    except Exception as e:
        print(f"[MemDbg][{tag}] mem report failed: {e}")

def _module_dtype_device(name: str, module: torch.nn.Module | None):
    if not MEM_DEBUG or module is None:
        return
    try:
        p = next(module.parameters())
        print(f"[MemDbg][Module] {name}: dtype={p.dtype} device={p.device} trainable_params={sum(int(q.requires_grad) for q in module.parameters())}")
    except StopIteration:
        print(f"[MemDbg][Module] {name}: no params")
    except Exception as e:
        print(f"[MemDbg][Module] {name}: inspect failed: {e}")

 

def _attach_backward_mem_hooks_for_blocks(pipe_module):
    if not MEM_DEBUG:
        return
    try:
        dit = getattr(pipe_module, 'dit', None)
        base = getattr(dit, 'base_model', dit)
        blocks = getattr(base, 'blocks', None)
        if blocks is None:
            return
        n = len(blocks)
        # Sample a few blocks across depth to avoid spam
        sample_idx = sorted(set([0, max(0, n//5), max(0, 2*n//5), max(0, 3*n//5), max(0, 4*n//5), n-1]))
        print(f"[MemDbg][Hooks] registering backward mem hooks on blocks {sample_idx}")
        def make_hook(idx):
            def hook(mod, grad_input, grad_output):
                _gpu_mem_report(f"bwd_block_{idx}")
            return hook
        for idx in sample_idx:
            try:
                blocks[idx].register_full_backward_hook(make_hook(idx))
            except Exception:
                pass
    except Exception as e:
        print(f"[MemDbg][Hooks] failed to register: {e}")



def enable_gc(model, on: bool = True, verbose: bool = True) -> bool:
    """Enable/disable gradient checkpointing on a (possibly PEFT-wrapped) model.
    Returns True if toggled, False otherwise.
    """
    try:
        base = getattr(model, 'base_model', model)
        if hasattr(base, 'enable_gradient_checkpointing') and on:
            base.enable_gradient_checkpointing()
            if verbose:
                print("[enable_gc] Gradient checkpointing enabled via method")
            return True
        if hasattr(base, 'gradient_checkpointing'):
            setattr(base, 'gradient_checkpointing', bool(on))
            if verbose:
                print(f"[enable_gc] Gradient checkpointing flag set to {bool(on)}")
            return True
        if verbose:
            print("[enable_gc] Model has no known GC toggle")
        return False
    except Exception as e:
        if verbose:
            print(f"[enable_gc] Failed to toggle GC: {e}")
        return False


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dataset_base_path=None,
        sf_restrict_timesteps=False,
        sf_denoising_step_list="1000,750,500,250",
        sf_warp_denoising_step=True,
        sf_timestep_shift=5.0,
        # External CausalWan integration (optional)
        use_causal_wan=False,
        causal_wan_model_file=None,
        causal_wan_config=None,
        causal_wan_kwargs=None,
        causal_wan_weights=None,
        causal_wan_lora_rank=None,
        causal_wan_lora_alpha=64.0,
        causal_wan_lora_targets="q,k,v,o,ffn.0,ffn.2",
        causal_wan_lora_init="kaiming",
        audio_frames_per_block: int = 3,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Replace DiT with external CausalWan if requested
        if use_causal_wan and causal_wan_model_file is not None:
            # Parse JSON kwargs if provided
            extra_kw = {}
            if causal_wan_kwargs is not None:
                try:
                    extra_kw = json.loads(causal_wan_kwargs)
                except Exception as e:
                    print(f"[CausalWan] Failed to parse causal_wan_kwargs JSON: {e}")
            try:
                self.pipe.load_causal_wan(
                    model_file=causal_wan_model_file,
                    config_path=causal_wan_config,
                    weights_path=causal_wan_weights,
                    adapter_weights_path=getattr(self, "causal_wan_adapter_weights", None),
                    lora_rank=causal_wan_lora_rank,
                    lora_alpha=causal_wan_lora_alpha,
                    lora_targets=causal_wan_lora_targets.split(',') if causal_wan_lora_targets else None,
                    lora_init=causal_wan_lora_init,
                    **extra_kw,
                )
                print("[CausalWan] Loaded external CausalWanModel and attached as pipe.dit")
            except Exception as e:
                print(f"[CausalWan] Failed to load external CausalWanModel: {e}")

        # Training mode
        # If we are using external causal WAN, avoid double-injecting LoRA via DiffSynth
        lora_base_model_effective = None if (use_causal_wan and causal_wan_model_file is not None) else lora_base_model
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model_effective, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        # Precomputed inputs toggles/keys
        self.use_precomputed_context = getattr(args, "use_precomputed_context", False)
        self.use_precomputed_latents = getattr(args, "use_precomputed_latents", False)
        self.precomputed_context_key = getattr(args, "precomputed_context_key", "context_path")
        self.precomputed_latents_key = getattr(args, "precomputed_latents_key", "vae_latents_path")
        # Memory knob for audio path block size
        setattr(self.pipe, 'audio_frames_per_block', int(audio_frames_per_block))
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.dataset_base_path = dataset_base_path
        # Configure optional Self-Forcing-style discrete timesteps
        self.pipe.sf_allowed_timestep_indices = None
        if sf_restrict_timesteps:
            # Ensure scheduler matches the shift used by Self-Forcing
            self.pipe.scheduler.set_timesteps(1000, training=True, shift=sf_timestep_shift)
            steps = [int(s) for s in sf_denoising_step_list.split(",") if s.strip()]
            if sf_warp_denoising_step:
                # timesteps[1000 - step] mapping
                indices = [1000 - s for s in steps]
            else:
                indices = steps
            # Clamp to valid range
            indices = [i for i in indices if 0 <= i < len(self.pipe.scheduler.timesteps)]
            self.pipe.sf_allowed_timestep_indices = torch.tensor(indices, dtype=torch.long)
            # Debug print: show the warped timestep indices and values
            try:
                vals = self.pipe.scheduler.timesteps[self.pipe.sf_allowed_timestep_indices]
                print("[SF] Restricted timestep indices:", self.pipe.sf_allowed_timestep_indices.tolist())
                print("[SF] Restricted timestep values:", [float(v) for v in vals])
            except Exception as e:
                print("[SF] Failed to print restricted timesteps:", e)

        # If reference image y is expected, upgrade DiT to accept 36 input channels (16 latents + 20 y)
        if "input_image" in self.extra_inputs:
            dit = self.pipe.dit
            if getattr(dit, "in_dim", 16) != 33:
                old_conv: torch.nn.Conv3d = dit.patch_embedding
                out_channels = old_conv.out_channels
                kT, kH, kW = old_conv.kernel_size
                stride = old_conv.stride
                padding = old_conv.padding
                dilation = old_conv.dilation
                bias_flag = old_conv.bias is not None
                # Create new conv with 36 input channels
                new_conv = torch.nn.Conv3d(33, out_channels, kernel_size=(kT, kH, kW), stride=stride, padding=padding, dilation=dilation, bias=bias_flag)
                # Zero init weights and copy old weights into the first 16 input channels
                with torch.no_grad():
                    new_conv.weight.zero_()
                    if bias_flag:
                        new_conv.bias.copy_(old_conv.bias)
                    new_conv.weight[:, :old_conv.in_channels, :, :, :].copy_(old_conv.weight)
                # Ensure dtype/device consistency with pipeline compute dtype
                new_conv = new_conv.to(dtype=self.pipe.torch_dtype)
                dit.patch_embedding = new_conv
                dit.in_dim = 33
                dit.require_vae_embedding = True

        # If audio embeddings are expected, ensure audio modules exist and are zero-initialized
        if "audio_emb" in self.extra_inputs:
            dit = self.pipe.dit
            if not hasattr(dit, "audio_proj") or dit.audio_proj is None:
                dit.audio_proj = AudioPack(in_channels=10752, patch_size=(4,1,1), dim=32, layernorm=True)
            if not hasattr(dit, "audio_cond_projs") or dit.audio_cond_projs is None:
                num_layers = len(dit.blocks)
                dit.audio_cond_projs = torch.nn.ModuleList([torch.nn.Linear(32, dit.dim) for _ in range(max(num_layers // 2 - 1, 0))])
            # Move to pipeline dtype for consistency
            dit.audio_proj = dit.audio_proj.to(dtype=self.pipe.torch_dtype)
            dit.audio_cond_projs = dit.audio_cond_projs.to(dtype=self.pipe.torch_dtype)
            # Set trainable and warm-up init to avoid zero-grad deadlock
            for p in dit.audio_proj.parameters():
                p.requires_grad = True
            # Warm AudioPack projection with small normal noise; keep bias zero
            if hasattr(dit.audio_proj, "proj"):
                if hasattr(dit.audio_proj.proj, "weight"):
                    torch.nn.init.normal_(dit.audio_proj.proj.weight, mean=0.0, std=1e-3)
                if hasattr(dit.audio_proj.proj, "bias") and dit.audio_proj.proj.bias is not None:
                    torch.nn.init.zeros_(dit.audio_proj.proj.bias)
            # Keep per-layer projections zero so forward starts as no-op but grads flow into them
            for lin in dit.audio_cond_projs:
                lin.weight.data.zero_()
                if lin.bias is not None:
                    lin.bias.data.zero_()
                for p in lin.parameters():
                    p.requires_grad = True
        
        # If using precomputed text embeddings, remove the prompt embedder unit to avoid overriding
        if self.use_precomputed_context:
            try:
                before_n = len(self.pipe.units)
                self.pipe.units = [u for u in self.pipe.units if u.__class__.__name__ != "WanVideoUnit_PromptEmbedder"]
                after_n = len(self.pipe.units)
                if MEM_DEBUG:
                    print(f"[MemDbg][Precomputed] Removed PromptEmbedder unit ({before_n}->{after_n})")
            except Exception:
                pass
        # If using precomputed latents, remove the ImageEmbedderVAE unit to avoid VAE usage
        if self.use_precomputed_latents:
            try:
                before_n = len(self.pipe.units)
                self.pipe.units = [u for u in self.pipe.units if u.__class__.__name__ != "WanVideoUnit_ImageEmbedderVAE"]
                after_n = len(self.pipe.units)
                if MEM_DEBUG:
                    print(f"[MemDbg][Precomputed] Removed ImageEmbedderVAE unit ({before_n}->{after_n})")
            except Exception:
                pass
        
    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Optionally load precomputed text embeddings and/or VAE latents
        # Text embeddings: expects a .pt with either the tensor directly or a dict containing 'prompt_embeds'
        if self.use_precomputed_context and self.precomputed_context_key in data and data[self.precomputed_context_key] is not None:
            path = data[self.precomputed_context_key]
            if isinstance(path, str):
                if not os.path.isabs(path) and self.dataset_base_path is not None:
                    path = os.path.join(self.dataset_base_path, path)
                try:
                    ctx = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    ctx = torch.load(path, map_location="cpu")
                if isinstance(ctx, dict):
                    ctx = ctx.get("prompt_embeds", ctx.get("context", ctx))
                if torch.is_tensor(ctx):
                    # Ensure shape [B, L, D]
                    if ctx.dim() == 2:
                        ctx = ctx.unsqueeze(0)
                    inputs_shared["context"] = ctx.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                    # Disable prompt to avoid unit recompute
                    inputs_posi.pop("prompt", None)
                else:
                    print(f"[Precomputed] Unexpected context payload at {path}; skipping")
        
        # VAE latents: expects .pt tensor shaped like [1, 16, T, H/8, W/8] or [T, 16, H/8, W/8] or [1, T, 16, H/8, W/8]
        if self.use_precomputed_latents and self.precomputed_latents_key in data and data[self.precomputed_latents_key] is not None:
            path = data[self.precomputed_latents_key]
            if isinstance(path, str):
                if not os.path.isabs(path) and self.dataset_base_path is not None:
                    path = os.path.join(self.dataset_base_path, path)
                try:
                    z = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    z = torch.load(path, map_location="cpu")
                if torch.is_tensor(z):
                    # Normalize to [1, 16, T, H, W]
                    if z.dim() == 4:  # [T, C, H, W] likely
                        if z.shape[1] == 16:
                            z = z.unsqueeze(0).permute(0, 2, 1, 3, 4)  # -> [1, 16, T, H, W]
                        else:
                            # [T, H, W, C] or other; try to guess not supported
                            pass
                    elif z.dim() == 5:
                        if z.shape[1] == 16:  # [B, 16, T, H, W]
                            pass
                        elif z.shape[2] == 16:  # [B, T, 16, H, W]
                            z = z.permute(0, 2, 1, 3, 4)
                        elif z.shape[0] == 16:  # [16, T, H, W, ?] unlikely
                            z = z.unsqueeze(0)
                    inputs_shared["input_latents"] = z.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                    # Also remove raw input video to avoid VAE encoding
                    inputs_shared["input_video"] = None
                    # Construct y (mask + reference) from first latent frame of precomputed latents (channels-first)
                    try:
                        z_btchw = inputs_shared["input_latents"]  # [1, 16, T, H, W]
                        if z_btchw.dim() != 5 or z_btchw.shape[1] != 16:
                            raise ValueError("input_latents shape must be [1, 16, T, H, W]")
                        b, c, t, h, w = z_btchw.shape
                        # Reference latent is first frame: [1, 16, 1, H, W] -> expand along T
                        ref_first = z_btchw[:, :, 0:1]                    # [1, 16, 1, H, W]
                        ref_expand = ref_first.repeat(1, 1, t, 1, 1)      # [1, 16, T, H, W]
                        # Mask channels-first: [1, 1, T, H, W] with first frame 1, others 0
                        mask_cf = torch.zeros((b, 1, t, h, w), dtype=self.pipe.torch_dtype, device=self.pipe.device)
                        mask_cf[:, :, 0] = 1
                        # Concatenate mask + reference along channel dim -> [1, 17, T, H, W]
                        y = torch.cat([mask_cf, ref_expand.to(dtype=self.pipe.torch_dtype)], dim=1)
                        inputs_shared["y"] = y
                        if MEM_DEBUG:
                            print(f"[MemDbg][Precomputed] Built y from precomputed latents: y={tuple(y.shape)}")
                    except Exception as e:
                        print(f"[Precomputed] Failed to build y from input_latents: {e}")
                else:
                    print(f"[Precomputed] Unexpected latents payload at {path}; skipping")
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "audio_emb":
                raw = data.get("audio_emb", None)
                if raw is None:
                    continue
                # Load tensor if a path string is provided
                if isinstance(raw, str):
                    path = raw
                    if not os.path.isabs(path) and hasattr(self, "dataset_base_path") and self.dataset_base_path is not None:
                        path = os.path.join(self.dataset_base_path, path)
                    try:
                        audio_emb = torch.load(path, map_location="cpu", weights_only=False)
                    except TypeError:
                        audio_emb = torch.load(path, map_location="cpu")
                    # If payload is a dict with audio tokens, pick that tensor
                    if isinstance(audio_emb, dict):
                        if "audio_tokens" in audio_emb:
                            audio_emb = audio_emb["audio_tokens"]
                        else:
                            for k in ("audio_emb", "audio", "tokens"):
                                if k in audio_emb:
                                    audio_emb = audio_emb[k]
                                    break
                else:
                    audio_emb = torch.as_tensor(raw)
                # Normalize to [1, L, D]
                if audio_emb.dim() == 2:
                    audio_emb = audio_emb.unsqueeze(0)
                elif audio_emb.dim() == 3 and audio_emb.shape[0] != 1:
                    audio_emb = audio_emb[:1]
                # Slice/pad to match first num_frames used by video loader
                target_len = inputs_shared["num_frames"]
                cur_len = audio_emb.shape[1]
                if cur_len >= target_len:
                    audio_emb = audio_emb[:, :target_len]
                else:
                    pad = torch.zeros(audio_emb.shape[0], target_len - cur_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                    audio_emb = torch.cat([audio_emb, pad], dim=1)
                inputs_shared["audio_emb"] = audio_emb
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
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
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dataset_base_path=args.dataset_base_path,
        sf_restrict_timesteps=args.sf_restrict_timesteps,
        sf_denoising_step_list=args.sf_denoising_step_list,
        sf_warp_denoising_step=args.sf_warp_denoising_step,
        sf_timestep_shift=args.sf_timestep_shift,
        use_causal_wan=getattr(args, "use_causal_wan", False),
        causal_wan_model_file=getattr(args, "causal_wan_model_file", None),
        causal_wan_config=getattr(args, "causal_wan_config", None),
        causal_wan_kwargs=getattr(args, "causal_wan_kwargs", None),
        causal_wan_weights=getattr(args, "causal_wan_weights", None),
        causal_wan_lora_rank=getattr(args, "causal_wan_lora_rank", None),
        causal_wan_lora_alpha=getattr(args, "causal_wan_lora_alpha", 64.0),
        causal_wan_lora_targets=getattr(args, "causal_wan_lora_targets", "q,k,v,o,ffn.0,ffn.2"),
        causal_wan_lora_init=getattr(args, "causal_wan_lora_init", "kaiming"),
        audio_frames_per_block=getattr(args, "audio_frames_per_block", 3),
    )
    # One-time dtype/device summary
    if MEM_DEBUG:
        try:
            pipe = model.pipe
            _module_dtype_device("text_encoder", getattr(pipe, "text_encoder", None))
            dit = getattr(pipe, "dit", None)
            if dit is not None and hasattr(dit, "base_model"):
                _module_dtype_device("dit.peft_base_model", dit.base_model)
            _module_dtype_device("dit", dit)
            _module_dtype_device("vae", getattr(pipe, "vae", None))
            # LoRA param count
            if dit is not None:
                lora_params = [(n, p) for n, p in dit.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
                total = sum(p.numel() for _, p in lora_params)
                total_bytes = sum(p.numel() * p.element_size() for _, p in lora_params)
                print(f"[MemDbg][LoRA] trainable lora params={total:,} ~{_bytes_to_gb(total_bytes):.3f}GB across {len(lora_params)} tensors")
            # Attach backward memory hooks on a subset of blocks
            _attach_backward_mem_hooks_for_blocks(pipe)
        except Exception as e:
            print(f"[MemDbg] module summary failed: {e}")
    # Optionally enable gradient checkpointing on the loaded DiT/CausalWan
    if getattr(args, "enable_gc", False):
        try:
            # Enable on primary model
            enable_gc(model.pipe.dit, on=True, verbose=True)
            # Enable on secondary if present
            if hasattr(model.pipe, 'dit2') and model.pipe.dit2 is not None:
                enable_gc(model.pipe.dit2, on=True, verbose=True)
        except Exception as e:
            print(f"[enable_gc] Failed to enable on pipeline models: {e}")
    # Pass adapter path to module instance if provided by CLI
    if hasattr(args, "causal_wan_adapter_weights"):
        setattr(model, "causal_wan_adapter_weights", getattr(args, "causal_wan_adapter_weights", None))
    # Reconstruct module with LoRA args (already passed above)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    # Custom training loop with accumulated/EMA loss logging to Weights & Biases.
    def launch_training_task_with_accum_logging(
        dataset,
        model,
        model_logger,
        args,
    ):
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters

        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
        accelerator = Accelerator(
            mixed_precision=getattr(args, "mixed_precision", None),
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        )
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

        # Quick precision/dtype sanity print (once)
        try:
            if accelerator.is_main_process:
                dit = model.pipe.dit
                p = next(dit.parameters())
                print(f"[Precision] accelerate.mixed_precision={accelerator.mixed_precision}; model param dtype={p.dtype}")
        except Exception:
            pass

        # Optional W&B setup
        use_wandb = getattr(args, "use_wandb", False)
        wandb_log_every = getattr(args, "wandb_log_every", 10)
        if use_wandb and accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=getattr(args, "wandb_project", "DiffSynth"),
                    entity=getattr(args, "wandb_entity", None),
                    name=getattr(args, "wandb_run_name", None),
                    tags=(getattr(args, "wandb_tags", None) or "").split(",") if getattr(args, "wandb_tags", None) else None,
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "num_epochs": num_epochs,
                    },
                )
            except Exception as e:
                use_wandb = False
                print(f"[W&B] Disabled due to import/init error: {e}")

        # Accumulation trackers (optimizer-step granularity)
        global_step = 0  # counts optimizer steps (after gradient accumulation)
        cum_loss_sum = 0.0
        cum_loss_count = 0
        window_loss_sum = 0.0
        window_loss_count = 0
        ema_loss = None
        ema_beta = 0.98
        # microstep accumulation for one optimizer step
        ga_loss_sum = 0.0
        ga_loss_count = 0

        for epoch_id in range(num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    if MEM_DEBUG:
                        torch.cuda.reset_peak_memory_stats()
                        _gpu_mem_report("step_begin", device=accelerator.device)
                    optimizer.zero_grad()
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        if MEM_DEBUG:
                            _gpu_mem_report("before_forward", device=accelerator.device)
                        loss = model(data)
                        if MEM_DEBUG:
                            _gpu_mem_report("after_forward", device=accelerator.device)
                    if MEM_DEBUG:
                        _gpu_mem_report("before_backward", device=accelerator.device)
                    try:
                        accelerator.backward(loss)
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[MemDbg][OOM] during backward: {e}")
                        _gpu_mem_report("oom_backward", device=accelerator.device)
                        raise
                    if MEM_DEBUG:
                        _gpu_mem_report("after_backward", device=accelerator.device)
                    optimizer.step()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    scheduler.step()
                    if MEM_DEBUG:
                        _gpu_mem_report("after_optimizer", device=accelerator.device)

                    # Convert to scalar safely on CPU to avoid blocking the graph
                    with torch.no_grad():
                        micro_loss = float(loss.detach().to("cpu", dtype=torch.float32))
                    ga_loss_sum += micro_loss
                    ga_loss_count += 1

                    # Only treat as a training "step" when gradients are synchronized
                    if accelerator.sync_gradients:
                        step_loss = ga_loss_sum / max(1, ga_loss_count)
                        ga_loss_sum = 0.0
                        ga_loss_count = 0

                        global_step += 1
                        cum_loss_sum += step_loss
                        cum_loss_count += 1
                        window_loss_sum += step_loss
                        window_loss_count += 1
                        ema_loss = step_loss if ema_loss is None else (ema_beta * ema_loss + (1 - ema_beta) * step_loss)

                        # Periodic logging (main process only)
                        if use_wandb and accelerator.is_main_process and (global_step % wandb_log_every == 0):
                            try:
                                import wandb
                                log = {
                                    "loss/ga_mean": step_loss,  # loss averaged inside one optimizer step
                                    "loss/ema": ema_loss,
                                    "loss/window_mean": window_loss_sum / max(1, window_loss_count),
                                    "loss/cum_mean": cum_loss_sum / max(1, cum_loss_count),
                                    "train/epoch": epoch_id,
                                    "train/step": global_step,
                                }
                                wandb.log(log, step=global_step)
                            except Exception as e:
                                # Non-fatal logging failure
                                print(f"[W&B] log error at step {global_step}: {e}")
                            # Reset window stats after logging
                            window_loss_sum = 0.0
                            window_loss_count = 0

            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

        if use_wandb and accelerator.is_main_process:
            try:
                import wandb
                wandb.summary["final/loss_ema"] = ema_loss
                wandb.finish()
            except Exception as e:
                print(f"[W&B] finish error: {e}")

        model_logger.on_training_end(accelerator, model, save_steps)

    launch_training_task_with_accum_logging(dataset, model, model_logger, args)
