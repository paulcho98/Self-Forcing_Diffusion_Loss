import torch

# Path to your original Self-Forcing checkpoint
pt_file_path = "/home/cvlab20/project/hyunbin/Self-Forcing/checkpoints/self_forcing_dmd.pt"

print(f"Loading checkpoint: {pt_file_path}")
data = torch.load(pt_file_path, map_location="cpu")

# --- Inspect Top-Level Structure ---
print("\n--- Top-Level Keys ---")
# This shows you all the main components saved in the file
print(list(data.keys()))

# --- Inspect Model Weights ---
if 'generator_ema' in data:
    print("\n--- Model Layer Names (from 'generator_ema') ---")
    state_dict = data['generator_ema']
    layer_names = list(state_dict.keys())
    
    # Print the first 10 layer names to see the naming convention
    for name in layer_names[:10]:
        print(f"- {name} | Shape: {state_dict[name].shape}")
    
    print(f"\nFound a total of {len(layer_names)} layers in the model.")
else:
    print("\nCould not find 'generator_ema' key. Inspect the top-level keys above to find the right one.")