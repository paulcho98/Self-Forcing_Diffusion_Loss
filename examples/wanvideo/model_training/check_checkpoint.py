import torch
import safetensors.torch

# --- 1. CONFIGURE YOUR CHECKPOINT PATHS HERE ---

file_a_path = "/home/cvlab20/project/jinhyuk/DiffSynth-Studio/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors"
file_b_path = "/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

# Set to True to compute the difference in tensor values, which can be slow.
COMPUTE_VALUE_DIFFERENCES = True

# --- 2. THE SCRIPT WILL HANDLE THE REST ---

def get_human_readable_size(num_bytes: int) -> str:
    """Converts a number of bytes to a human-readable string."""
    if num_bytes is None: return "N/A"
    if num_bytes < 1024: return f"{num_bytes} B"
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num_bytes) < 1024.0: return f"{num_bytes:3.1f} {unit}B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"

def analyze_checkpoint(file_path: str):
    """Loads a checkpoint and returns its state dict and metadata."""
    try:
        state_dict = safetensors.torch.load_file(file_path, device="cpu")
        total_params = sum(p.numel() for p in state_dict.values())
        total_size_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())
        return state_dict, {
            "path": file_path, "params": total_params, "size": get_human_readable_size(total_size_bytes)
        }
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        exit()
    except Exception as e:
        print(f"❌ Error loading file '{file_path}': {e}")
        exit()

def compare_checkpoints():
    """Main function to load and compare two checkpoints."""
    print("--- 🕵️ Explicit Checkpoint Comparison Tool ---")

    # Load checkpoints
    print(f"\nLoading Checkpoint A: {file_a_path}")
    state_dict_a, meta_a = analyze_checkpoint(file_a_path)
    print(f"Loading Checkpoint B: {file_b_path}")
    state_dict_b, meta_b = analyze_checkpoint(file_b_path)

    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())
    common_keys = sorted(list(keys_a & keys_b))

    # --- Overall Summary ---
    print("\n\n--- ## Overall Summary ## ---")
    print(f"Checkpoint A: {len(keys_a)} tensors | {meta_a['params']:,} parameters | Size: {meta_a['size']}")
    print(f"Checkpoint B: {len(keys_b)} tensors | {meta_b['params']:,} parameters | Size: {meta_b['size']}")
    print("-" * 30)

    # --- Layer Name (Key) Differences ---
    print("\n--- ## Layer Name Differences ## ---")
    unique_to_a = sorted(list(keys_a - keys_b))
    unique_to_b = sorted(list(keys_b - keys_a))
    if not unique_to_a and not unique_to_b:
        print("✅ Layer names are identical.")
    else:
        if unique_to_a:
            print(f"⚠️ {len(unique_to_a)} layers ONLY in Checkpoint A (showing up to 5): {unique_to_a[:5]}")
        if unique_to_b:
            print(f"⚠️ {len(unique_to_b)} layers ONLY in Checkpoint B (showing up to 5): {unique_to_b[:5]}")
    print("-" * 30)

    # --- Explicit Checks for Shared Layers ---
    shape_mismatches = []
    dtype_mismatches = []
    value_mismatches = []

    for key in common_keys:
        t_a = state_dict_a[key]
        t_b = state_dict_b[key]
        
        # 1. Check Shape
        if t_a.shape != t_b.shape:
            shape_mismatches.append(f"Layer '{key}': A is {t_a.shape}, B is {t_b.shape}")
            continue # No need to check dtype or value if shapes differ
        
        # 2. Check Data Type
        if t_a.dtype != t_b.dtype:
            dtype_mismatches.append(f"Layer '{key}': A is {t_a.dtype}, B is {t_b.dtype}")
            continue # No need to check value if dtypes differ

        # 3. Check Values
        if COMPUTE_VALUE_DIFFERENCES:
            if not torch.equal(t_a, t_b):
                abs_diff = torch.mean(torch.abs(t_a.float() - t_b.float())).item()
                value_mismatches.append(f"Layer '{key}': Values differ (Avg. diff: {abs_diff:.6f})")

    # --- Print Structured Report ---
    print("\n--- ## Explicit Shape (Size) Comparison ## ---")
    if not shape_mismatches:
        print(f"✅ No shape mismatches found across all {len(common_keys)} shared layers.")
    else:
        print(f"❌ Found {len(shape_mismatches)} layers with shape mismatches:")
        for error in shape_mismatches[:10]: print(f"  - {error}")
        if len(shape_mismatches) > 10: print("  ...")
    print("-" * 30)

    print("\n--- ## Explicit Data Type (Dtype) Comparison ## ---")
    if not dtype_mismatches:
        print(f"✅ No data type mismatches found across all {len(common_keys)} shared layers.")
    else:
        print(f"❌ Found {len(dtype_mismatches)} layers with data type mismatches:")
        for error in dtype_mismatches[:10]: print(f"  - {error}")
        if len(dtype_mismatches) > 10: print("  ...")
    print("-" * 30)

    if COMPUTE_VALUE_DIFFERENCES:
        print("\n--- ## Explicit Value Comparison ## ---")
        if not value_mismatches:
            print(f"✅ All tensor values are identical across all {len(common_keys)} shared layers.")
        else:
            print(f"⚠️ Found {len(value_mismatches)} layers with different numerical values:")
            for error in value_mismatches[:10]: print(f"  - {error}")
            if len(value_mismatches) > 10: print("  ...")
        print("-" * 30)
    
    print("\n--- ✅ Comparison Complete ---")

if __name__ == "__main__":
    compare_checkpoints()