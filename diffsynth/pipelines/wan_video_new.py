import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import torch.nn as nn

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.audio_pack import AudioPack
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader

# Note: Do NOT statically import Self-Forcing CausalWanModel here.
# This pipeline loads it dynamically in load_causal_wan(), where sys.path
# is adjusted to include the parent that contains the 'wan' package.


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.kv_cache = None
        self.crossattn_cache = None
        
    def _mem_debug_enabled(self) -> bool:
        try:
            if getattr(self, 'mem_debug', False):
                return True
        except Exception:
            pass
        return os.environ.get('MEM_DEBUG', '0') == '1'

    def _bytes_to_gb(self, x: int | float) -> float:
        try:
            return float(x) / (1024 ** 3)
        except Exception:
            return float(x)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache = []

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache = kv_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = kv_cache[0]["k"].numel() * kv_cache[0]["k"].element_size() + \
                                   kv_cache[0]["v"].numel() * kv_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(kv_cache)
                print(f"[MemDbg][KV] layers={len(kv_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.2f}GB total={self._bytes_to_gb(total_bytes):.2f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][KV] calc failed: {e}")

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = crossattn_cache[0]["k"].numel() * crossattn_cache[0]["k"].element_size() + \
                                   crossattn_cache[0]["v"].numel() * crossattn_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(crossattn_cache)
                print(f"[MemDbg][XATTN] layers={len(crossattn_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.3f}GB total={self._bytes_to_gb(total_bytes):.3f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][XATTN] calc failed: {e}")

    def load_causal_wan(
        self,
        model_file: str,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        adapter_weights_path: Optional[str] = None,
        use_ema: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 64.0,
        lora_targets: Optional[list[str]] = None,
        lora_init: str = "kaiming",
        **kwargs,
    ):
        """Dynamically load an external CausalWanModel from a local python file.
        - model_file: filesystem path to causal_model.py that defines CausalWanModel
        - config_path: optional path to JSON config with base args
        - kwargs: override arguments (e.g., use_audio=True, in_dim=33, audio_hidden_size=32)
        """
        import json, importlib.util, sys
        from pathlib import Path
        # Import module from file
        # Ensure the repository root (containing the 'wan' package) is on sys.path
        try:
            mf = Path(model_file).resolve()
            # heuristically climb up to find the folder that contains 'wan'
            add_path = None
            for parent in [mf.parent, *mf.parents]:
                if (parent / 'wan').exists() and (parent / 'wan').is_dir():
                    add_path = str(parent)
                    break
            if add_path and add_path not in sys.path:
                sys.path.insert(0, add_path)
        except Exception:
            pass

        spec = importlib.util.spec_from_file_location("external_causal_wan", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {model_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "CausalWanModel"):
            raise AttributeError(f"CausalWanModel not found in {model_file}")
        CausalWanModel = getattr(mod, "CausalWanModel")

        # Load base config if provided
        base_cfg = {}
        if config_path is not None:
            try:
                with open(config_path, "r") as f:
                    base_cfg = json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to read CausalWan config at {config_path}: {e}")

        # Merge kwargs over base config
        init_kwargs = dict(base_cfg)
        init_kwargs.update(kwargs or {})
        # Special flags not part of CausalWanModel ctor
        zero_audio_proj_flag = bool(init_kwargs.pop('zero_audio_proj', False))

        # Heuristically align constructor dims with checkpoint if provided
        if weights_path is not None:
            try:
                if weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safe_load
                    raw_sd = safe_load(weights_path)
                else:
                    raw_sd = torch.load(weights_path, map_location='cpu')
                # unwrap common containers
                if isinstance(raw_sd, dict):
                    for key in (['generator_ema', 'ema'] if use_ema else []) + ['generator','model','state_dict','module','student','net']:
                        if key in raw_sd and isinstance(raw_sd[key], dict):
                            raw_sd = raw_sd[key]
                            break
                # Try to infer dim and num_layers from ckpt
                ckpt_dim = None
                pe_key = 'model.patch_embedding.weight'
                if isinstance(raw_sd, dict) and pe_key in raw_sd and hasattr(raw_sd[pe_key], 'shape'):
                    ckpt_dim = int(raw_sd[pe_key].shape[0])
                ckpt_layers = None
                if isinstance(raw_sd, dict):
                    import re
                    layer_indices = []
                    for k in raw_sd.keys():
                        m = re.match(r"model\.blocks\.(\d+)\.", k)
                        if m:
                            try:
                                layer_indices.append(int(m.group(1)))
                            except Exception:
                                pass
                    if layer_indices:
                        ckpt_layers = max(layer_indices) + 1
                updated = False
                if ckpt_dim is not None and init_kwargs.get('dim', None) != ckpt_dim:
                    init_kwargs['dim'] = ckpt_dim
                    updated = True
                if ckpt_layers is not None and init_kwargs.get('num_layers', None) != ckpt_layers:
                    init_kwargs['num_layers'] = ckpt_layers
                    updated = True
                # Heuristic defaults for known CausalWan 1.3B
                if init_kwargs.get('dim', None) == 1536:
                    init_kwargs.setdefault('num_heads', 12)
                    init_kwargs.setdefault('ffn_dim', 8960)
                if updated:
                    print(f"[CausalWan] Heuristic init from ckpt: dim={init_kwargs.get('dim')} num_layers={init_kwargs.get('num_layers')} num_heads={init_kwargs.get('num_heads')} ffn_dim={init_kwargs.get('ffn_dim')}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to infer arch from weights: {e}")

        # Instantiate
        model = CausalWanModel(**init_kwargs)

        # Optional: apply LoRA via PEFT BEFORE loading weights (to match checkpoint structure)
        if lora_rank is not None and lora_rank > 0:
            try:
                from peft import LoraConfig, get_peft_model
                target_modules = lora_targets or ["q", "k", "v", "o", "ffn.0", "ffn.2"]
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    init_lora_weights=True,
                )
                model = get_peft_model(model, lora_cfg)
                # Freeze base model weights like Self-Forcing
                try:
                    for p in model.base_model.parameters():
                        p.requires_grad = False
                except Exception:
                    pass
                print(f"[CausalWan] Applied PEFT LoRA pre-load: r={lora_rank}, alpha={lora_alpha}, targets={target_modules}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to apply PEFT LoRA pre-load: {e}")

        # Load weights if provided (after LoRA) 
        if weights_path is not None:
            try:
                if weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safe_load
                    state = safe_load(weights_path)
                else:
                    state = torch.load(weights_path, map_location='cpu')
                # Common wrappers
                if isinstance(state, dict):
                    # Prefer the same priority used in Self-Forcing trainer
                    wrapper_order = []
                    if use_ema:
                        wrapper_order += ['generator_ema', 'ema']
                    wrapper_order += ['generator', 'model', 'state_dict', 'module', 'student', 'net']
                    for key in wrapper_order:
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                # Strip DistributedDataParallel and FSDP wrappers
                def strip_wrappers(name: str) -> str:
                    return name.replace('module.', '').replace('_checkpoint_wrapped_module.', '')
                state = {strip_wrappers(k): v for k, v in state.items()}

                # If PEFT-LoRA is requested, adapt checkpoint key space to PEFT structure
                def adapt_for_peft(sd: dict, target_in_dim: int, enable_lora: bool) -> dict:
                    import re
                    out = {}
                    for k, v in sd.items():
                        nk = k
                        if enable_lora:
                            # Map raw 'model.*' ckpt keys to PEFT namespace 'base_model.model.*'
                            if nk.startswith('model.'):
                                nk = 'base_model.model.' + nk[len('model.') :]
                            # If keys were already remapped to 'base_model.*', ensure 'base_model.model.*'
                            elif nk.startswith('base_model.') and not nk.startswith('base_model.model.'):
                                nk = 'base_model.model.' + nk[len('base_model.') :]
                        else:
                            # Non-PEFT: drop leading 'model.' if present
                            if nk.startswith('model.'):
                                nk = nk[len('model.') :]
                        out[nk] = v
                    # map leaf weights to .base_layer for LoRA-targets
                    if enable_lora:
                        mapped = {}
                        pats = [
                            re.compile(r"\.self_attn\.(q|k|v|o)\.(weight|bias)$"),
                            re.compile(r"\.cross_attn\.(q|k|v|o)\.(weight|bias)$"),
                            re.compile(r"\.ffn\.(0|2)\.(weight|bias)$"),
                        ]
                        for k, v in out.items():
                            mk = k
                            for pat in pats:
                                if pat.search(k) and '.base_layer.' not in k and '.lora_' not in k:
                                    head, leaf = k.rsplit('.', 1)
                                    mk = f"{head}.base_layer.{leaf}"
                                    break
                            mapped[mk] = v
                        out = mapped
                    # patch_embedding expansion when moving from 16->33 channels
                    if target_in_dim == 33:
                        candidate = 'base_model.model.patch_embedding.weight' if enable_lora else 'patch_embedding.weight'
                        if candidate in out:
                            w = out[candidate]
                            if hasattr(w, 'ndim') and w.ndim == 5 and w.shape[1] == 16:
                                expanded = torch.zeros(w.shape[0], 33, w.shape[2], w.shape[3], w.shape[4], dtype=w.dtype)
                                expanded[:, :16] = w
                                out[candidate] = expanded
                    return out

                target_in_dim = init_kwargs.get('in_dim', 16)
                enable_lora = (lora_rank is not None and lora_rank > 0)
                state_adapted = adapt_for_peft(state, target_in_dim=target_in_dim, enable_lora=enable_lora)

                missing, unexpected = model.load_state_dict(state_adapted, strict=False)
                if len(missing) > 0:
                    warnings.warn(f"[CausalWan] Missing keys when loading weights: {len(missing)}")
                if len(unexpected) > 0:
                    warnings.warn(f"[CausalWan] Unexpected keys when loading weights: {len(unexpected)}")
                print(f"[CausalWan] Loaded weights from {weights_path}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to load weights from {weights_path}: {e}")

        # Optionally load adapter (LoRA/audio) weights saved by our training loop (trainable-only SD)
        if adapter_weights_path is not None:
            try:
                if adapter_weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safe_load
                    adapter = safe_load(adapter_weights_path)
                else:
                    adapter = torch.load(adapter_weights_path, map_location='cpu')
                if isinstance(adapter, dict):
                    # Unwrap common containers
                    for key in ['generator', 'model', 'state_dict', 'module']:
                        if key in adapter and isinstance(adapter[key], dict):
                            adapter = adapter[key]
                            break
                # Map to PEFT namespace when LoRA is active
                target_in_dim = init_kwargs.get('in_dim', 16)
                enable_lora = (lora_rank is not None and lora_rank > 0)
                adapter_adapted = adapt_for_peft(adapter, target_in_dim=target_in_dim, enable_lora=enable_lora)
                missing_a, unexpected_a = model.load_state_dict(adapter_adapted, strict=False)
                if len(missing_a) > 0:
                    warnings.warn(f"[CausalWan] Missing keys when loading adapter: {len(missing_a)}")
                if len(unexpected_a) > 0:
                    warnings.warn(f"[CausalWan] Unexpected keys when loading adapter: {len(unexpected_a)}")
                print(f"[CausalWan] Loaded adapter weights from {adapter_weights_path}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to load adapter weights from {adapter_weights_path}: {e}")

        # Make audio modules trainable and warm-init similar to Self-Forcing
        base = getattr(model, 'base_model', model)
        audio_proj = getattr(base, 'audio_proj', None)
        audio_cond_projs = getattr(base, 'audio_cond_projs', None)
        # Ensure flags expected by DiffSynth units exist on the model (CausalWanModel doesn't define them by default)
        try:
            # Require VAE embedding when using 33 input channels (x+y)
            if not hasattr(base, 'require_vae_embedding'):
                setattr(base, 'require_vae_embedding', bool(init_kwargs.get('in_dim', 16) == 33))
            # Default: no CLIP image embedding path for CausalWanModel
            if not hasattr(base, 'require_clip_embedding'):
                setattr(base, 'require_clip_embedding', False)
            # Default: no special image positional embedding
            if not hasattr(base, 'has_image_pos_emb'):
                setattr(base, 'has_image_pos_emb', False)
            if not hasattr(base,'fuse_vae_embedding_in_latents'):
                setattr(base,'fuse_vae_embedding_in_latents', False)
        except Exception:
            pass
        try:
            print(
                f"[DBG] load_causal_wan: in_dim={getattr(base,'in_dim',None)} require_vae={getattr(base,'require_vae_embedding',None)} "
                f"require_clip={getattr(base,'require_clip_embedding',None)} has_image_pos_emb={getattr(base,'has_image_pos_emb',None)}",
                flush=True,
            )
        except Exception:
            pass
        if audio_proj is not None:
            proj = getattr(audio_proj, 'proj', None)
            if proj is not None:
                if hasattr(proj, 'weight') and proj.weight is not None:
                    with torch.no_grad():
                        if zero_audio_proj_flag:
                            proj.weight.zero_()
                        else:
                            torch.nn.init.normal_(proj.weight, mean=0.0, std=1e-3)
                if hasattr(proj, 'bias') and proj.bias is not None:
                    with torch.no_grad():
                        proj.bias.zero_()
            for p in audio_proj.parameters():
                p.requires_grad = True
        if audio_cond_projs is not None:
            for lin in audio_cond_projs:
                if hasattr(lin, 'weight') and lin.weight is not None:
                    with torch.no_grad():
                        lin.weight.zero_()
                if hasattr(lin, 'bias') and lin.bias is not None:
                    with torch.no_grad():
                        lin.bias.zero_()
                for p in lin.parameters():
                    p.requires_grad = True
        try:
            model = model.eval().to(dtype=self.torch_dtype, device=self.device)
        except Exception:
            model = model.to(device=self.device)
            try:
                model = model.to(dtype=self.torch_dtype)
            except Exception:
                pass

        # Initialize inference cache configuration based on the loaded model
        try:
            base = getattr(model, 'base_model', model)
            self.num_transformer_blocks = len(getattr(base, 'blocks'))
            self.frame_seq_length = 1560
            if getattr(base, 'local_attn_size', -1) != -1:
                self.kv_cache_size = int(base.local_attn_size) * self.frame_seq_length
            else:
                self.kv_cache_size = 32760
        except Exception:
            # Fallbacks
            self.num_transformer_blocks = getattr(self, 'num_transformer_blocks', 30)
            self.frame_seq_length = getattr(self, 'frame_seq_length', 1560)
            self.kv_cache_size = getattr(self, 'kv_cache_size', 32760)
        self.dit = model
        if self._mem_debug_enabled():
            try:
                base = getattr(model, 'base_model', model)
                p = next(base.parameters())
                print(f"[MemDbg][CausalWan] layers={self.num_transformer_blocks} frame_seq_len={self.frame_seq_length} kv_cache_size={self.kv_cache_size} model_dtype={p.dtype} device={p.device}")
            except Exception as e:
                print(f"[MemDbg][CausalWan] inspect failed: {e}")
        return model

    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        # Prefer restricted timesteps if provided (Self-Forcing compatibility)
        if hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None and len(self.sf_allowed_timestep_indices) > 0:
            idx = self.sf_allowed_timestep_indices[torch.randint(0, len(self.sf_allowed_timestep_indices), (1,))]
            timestep = self.scheduler.timesteps[idx].to(dtype=self.torch_dtype, device=self.device)
        else:
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        # Route to OmniAvatar-style audio path if explicit audio embeddings are provided.
        if inputs.get("audio_emb", None) is not None:
            try:
                y_dbg = inputs.get("y")
                print(
                    f"[DBG] training_loss: latents={tuple(inputs['latents'].shape)}, input_latents={tuple(inputs['input_latents'].shape)}, "
                    f"y={(tuple(y_dbg.shape) if y_dbg is not None else None)}, dit.in_dim={getattr(self.dit,'in_dim',None)}",
                    flush=True,
                )
            except Exception:
                pass
            noise_pred = self.model_fn_audio_new(
                dit=self.dit,
                latents=inputs["latents"],
                clean_latents=inputs["input_latents"],
                timestep=timestep,
                context=inputs["context"],
                y=inputs.get("y"),
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
                audio_emb=inputs.get("audio_emb"),
            )
        else:
            noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def model_fn_audio_new(
        self,
        dit: nn.Module, # WanModel
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """OmniAvatar-style audio conditioning with reference y, ignoring clip_feature.
        Expects `audio_emb` shaped [B, L, 10752] (or [L, 10752]).
        Uses AudioPack(t=4) and injects per-layer early residuals before transformer blocks.
        """
        assert audio_emb is not None, "audio_emb must be provided for model_fn_audio."
        # Latents are shaped [B, C, T, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape
        # Ensure audio embeddings are on same device/dtype as latents
        if audio_emb is not None:
            audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)
        try:
            print(
                f"[DBG] audio_new: dit.in_dim={getattr(dit,'in_dim',None)}, require_vae={getattr(dit,'require_vae_embedding',None)}, "
                f"latents={latents.shape}, y={(y.shape if y is not None else None)}",
                flush=True,
            )
        except Exception:
            pass
        frame_seq_length = getattr(self, 'frame_seq_length', 1560)  # tokens per frame
        # Allow tuning frames per block via pipeline attribute set by training module
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block  # Ceiling division

        # Initialize external caches for causal inference path
        self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
        self._initialize_crossattn_cache(batch_size, latents.dtype, latents.device)

        # Propagate GC/offload intent into the (possibly PEFT-wrapped) CausalWanModel
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            # Offload saved tensors to CPU if requested
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass
        if self._mem_debug_enabled():
            try:
                print(f"[MemDbg][AudioPath] latents={tuple(latents.shape)} dtype={latents.dtype} context={tuple(context.shape) if context is not None else None} audio_emb={tuple(audio_emb.shape) if audio_emb is not None else None}")
                dev = latents.device
                alloc = torch.cuda.memory_allocated(dev)
                reserv = torch.cuda.memory_reserved(dev)
                print(f"[MemDbg][AudioPath] after_cache_alloc: allocated={self._bytes_to_gb(alloc):.2f}GB reserved={self._bytes_to_gb(reserv):.2f}GB")
            except Exception as e:
                print(f"[MemDbg][AudioPath] inspect failed: {e}")

        output = torch.zeros(
                [batch_size, num_channels, num_frames, height, width],
                device=latents.device,
                dtype=latents.dtype
            )

        # First frame along temporal axis
        first_frame = clean_latents[:, :, :1]

        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx
            # Slice frames along temporal dimension: [B, C, Fblk, H, W]
            noisy_input = latents[:, :, start_idx:end_idx]
            if block_index == 0:
                noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]

            # Prepare y block: [B, Cy, Tzip, H, W] → [B, Cy, Fblk, H, W]
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
                # Concat along channel dim
                x_concat = torch.cat([noisy_input, y_block], dim=1)
            else:
                y_block = None
                x_concat = noisy_input
            try:
                xC = noisy_input.shape[1]
                yC = 0 if y_block is None else y_block.shape[1]
                exp = getattr(dit, 'in_dim', None)
                print(
                    f"[DBG] audio_new block={block_index} frames={cur_frames} start={start_idx} end={end_idx} "
                    f"xC={xC} yC={yC} total={xC+yC} exp_in_dim={exp}",
                    flush=True,
                )
            except Exception:
                pass

            # Build a per-batch per-frame timestep tensor: shape [B, Fblk]
            if timestep.numel() == 1:
                t_scalar = float(timestep.detach().float().item())
                t_block = torch.full((batch_size, cur_frames), t_scalar, device=latents.device, dtype=torch.float32)
            else:
                # If a vector was provided, broadcast or slice to [B, Fblk]
                t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
                if t_vec.numel() == batch_size:
                    t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
                else:
                    # Fallback: repeat scalar first element
                    t_block = torch.full((batch_size, cur_frames), float(t_vec[0].item()), device=latents.device, dtype=torch.float32)

            # Current start (unused without external caches)
            current_start_tokens = int(current_start_frame) * int(frame_seq_length)

            # Inference with caches
            denoised_pred = dit(
                x_concat,  # already [B, C, F, H, W]
                t=t_block,
                context=context,
                audio_emb=audio_emb,
                seq_len=cur_frames * frame_seq_length,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_tokens,
                **kwargs,
            )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] block_out={tuple(denoised_pred.shape)}")
                    print(f"[MemDbg][AudioPath] after_block: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.2: record the model's output along temporal dimension
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 2.3: update cache with clean latents
            t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
            clean_block = clean_latents[:, :, start_idx:end_idx]
            if block_index == 0:
                clean_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            if y is not None:
                clean_concat = torch.cat([clean_block, y_block], dim=1)
            else:
                clean_concat = clean_block
            with torch.no_grad():
                _ = dit(
                    clean_concat,
                    t=t_ctx,
                    context=context,
                    audio_emb=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] after_clean_ctx: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.4: update the start and end frame indices
            current_start_frame += cur_frames
        output[:, :, 0, :, :] = first_frame
        return output


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video

def _ensure_omni_audio_modules(dit: WanModel, audio_hidden_size: int = 32):
    """Lazy-init OmniAvatar-style audio modules on a WanModel instance.
    - AudioPack projects concatenated wav2vec features with temporal patching (t=4)
    - Per-layer linear projections map audio hidden to model dim for early blocks
    """
    if not hasattr(dit, "audio_proj") or dit.audio_proj is None:
        dit.audio_proj = AudioPack(in_channels=10752, patch_size=(4, 1, 1), dim=audio_hidden_size, layernorm=True)
    if not hasattr(dit, "audio_cond_projs") or dit.audio_cond_projs is None:
        num_layers = len(dit.blocks)
        num_inject = max(num_layers // 2 - 1, 0)
        dit.audio_cond_projs = torch.nn.ModuleList([
            torch.nn.Linear(audio_hidden_size, dit.dim) for _ in range(num_inject)
        ])
    dit.use_audio = True


def model_fn_audio(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    audio_emb: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """OmniAvatar-style audio conditioning with reference y, ignoring clip_feature.
    Expects `audio_emb` shaped [B, L, 10752] (or [L, 10752]).
    Uses AudioPack(t=4) and injects per-layer early residuals before transformer blocks.
    """
    assert audio_emb is not None, "audio_emb must be provided for model_fn_audio."

    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    # Text embedding
    context = dit.text_embedding(context)

    # Prepare input latents, fuse VAE image embedding y
    x = latents
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    # Record latent grid for audio spatial expansion
    lat_h, lat_w = latents.shape[-2], latents.shape[-1]

    # Patchify
    x, (f, h, w) = dit.patchify(x)

    # RoPE freqs for attention
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Prepare audio embeddings
    _ensure_omni_audio_modules(dit)
    if audio_emb.dim() == 2:
        audio_emb = audio_emb.unsqueeze(0)
    if audio_emb.shape[0] != x.shape[0]:
        audio_emb = audio_emb[:1].repeat(x.shape[0], 1, 1)
    audio_emb = audio_emb.to(device=x.device, dtype=x.dtype)
    # [B, 10752, L, 1, 1]
    audio_vid = audio_emb.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    # Optionally prepend a few frames to align with latent packing
    audio_vid = torch.cat([audio_vid[:, :, :1].repeat(1, 1, 3, 1, 1), audio_vid], dim=2)
    # AudioPack -> [B, T', 1, 1, H]
    audio_feat = dit.audio_proj(audio_vid)
    # Per-layer projection stack -> cat along a pseudo layer batch dim
    if len(dit.audio_cond_projs) > 0:
        audio_proj_stack = torch.concat([proj(audio_feat) for proj in dit.audio_cond_projs], dim=0)
        # Reshape to [B, LAYERS, T', 1, 1, dim]
        audio_proj_stack = audio_proj_stack.reshape(
            x.shape[0], audio_proj_stack.shape[0] // x.shape[0], audio_proj_stack.shape[1],
            audio_proj_stack.shape[2], audio_proj_stack.shape[3], audio_proj_stack.shape[4]
        )
    else:
        audio_proj_stack = None

    # Grid size for token alignment (spatial tokens per frame)
    tokens_h = max(lat_h // max(dit.patch_size[1], 1), 1)
    tokens_w = max(lat_w // max(dit.patch_size[2], 1), 1)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    num_layers = len(dit.blocks)
    for layer_i, block in enumerate(dit.blocks):
        # Audio injection into early blocks (2..num_layers//2), before transformer block
        if audio_proj_stack is not None and (layer_i <= num_layers // 2 and layer_i > 1):
            au_idx = layer_i - 2
            if 0 <= au_idx < audio_proj_stack.shape[1]:
                a = audio_proj_stack[:, au_idx]  # [B, T', 1, 1, dim]
                a = a.repeat(1, 1, tokens_h, tokens_w, 1)  # [B, T', H, W, dim]
                a_tokens = a.view(a.shape[0], -1, a.shape[-1])  # [B, (T'*H*W), dim]
                # Align lengths if necessary
                if a_tokens.shape[1] < x.shape[1]:
                    pad = x.shape[1] - a_tokens.shape[1]
                    a_tokens = torch.cat([a_tokens, torch.zeros(a_tokens.shape[0], pad, a_tokens.shape[2], device=a_tokens.device, dtype=a_tokens.dtype)], dim=1)
                else:
                    a_tokens = a_tokens[:, :x.shape[1]]
                x = x + a_tokens

        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    # Head and unpatchify
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


    




class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        try:
            print(f"[DBG-X][InputVideo] input_latents={tuple(input_latents.shape)}", flush=True)
        except Exception:
            pass
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        try:
            print(f"[DBG-Y][ImageDeprecated] y={tuple(y.shape)} (deprecated path)", flush=True)
        except Exception:
            pass
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        try:
            print(f"[DBG-Y][ImageCLIP] clip_feature={tuple(clip_context.shape)}", flush=True)
        except Exception:
            pass
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        # Prepare VAE input: first frame + zeros (optionally end image at last frame)
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 2, height, width, device=image.device, dtype=image.dtype),
                end_image.transpose(0, 1)
            ], dim=1)
        else:
            vae_input = torch.concat([
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width, device=image.device, dtype=image.dtype)
            ], dim=1)

        # Encode to 16-channel latents across zipped temporal length
        y_lat = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y_lat = y_lat.to(dtype=pipe.torch_dtype, device=pipe.device)
        # Build single-channel mask directly at post-VAE resolution: 0 at first step, 1 afterwards
        _, t_zip, h_lat, w_lat = y_lat.shape
        mask_zip = torch.ones(1, t_zip, h_lat, w_lat, device=pipe.device, dtype=pipe.torch_dtype)
        mask_zip[:, 0:1] = 0

        # Concatenate 1-channel mask with 16-channel latents => 17 channels total
        y = torch.cat([mask_zip, y_lat], dim=0).unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        try:
            print(f"[DBG-Y][ImageVAE] y={tuple(y.shape)} (mask+latents) maskC=1 latC=16", flush=True)
        except Exception:
            pass
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        try:
            print(
                f"[DBG-Y][FunControl] control_latentsC={control_latents.shape[1]} y_dim_after={y.shape[1]} expected_additional={(pipe.dit.in_dim - latents.shape[1])}",
                flush=True,
            )
        except Exception:
            pass
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        try:
            print(
                f"[DBG-Y][Camera] final y={tuple(y.shape)} expected_yC={(pipe.dit.in_dim - latents.shape[1])}",
                flush=True,
            )
        except Exception:
            pass
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio"), inputs_shared.pop("audio_embeds"), inputs_shared.get("audio_sample_rate")
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video"), inputs_shared.pop("s2v_pose_latents"), inputs_shared.pop("motion_video")

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
