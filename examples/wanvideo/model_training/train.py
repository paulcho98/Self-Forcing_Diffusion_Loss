import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.audio_pack import AudioPack
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



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
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
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
            if getattr(dit, "in_dim", 16) != 36:
                old_conv: torch.nn.Conv3d = dit.patch_embedding
                out_channels = old_conv.out_channels
                kT, kH, kW = old_conv.kernel_size
                stride = old_conv.stride
                padding = old_conv.padding
                dilation = old_conv.dilation
                bias_flag = old_conv.bias is not None
                # Create new conv with 36 input channels
                new_conv = torch.nn.Conv3d(36, out_channels, kernel_size=(kT, kH, kW), stride=stride, padding=padding, dilation=dilation, bias=bias_flag)
                # Zero init weights and copy old weights into the first 16 input channels
                with torch.no_grad():
                    new_conv.weight.zero_()
                    if bias_flag:
                        new_conv.bias.copy_(old_conv.bias)
                    new_conv.weight[:, :old_conv.in_channels, :, :, :].copy_(old_conv.weight)
                # Ensure dtype/device consistency with pipeline compute dtype
                new_conv = new_conv.to(dtype=self.pipe.torch_dtype)
                dit.patch_embedding = new_conv
                dit.in_dim = 36
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
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
