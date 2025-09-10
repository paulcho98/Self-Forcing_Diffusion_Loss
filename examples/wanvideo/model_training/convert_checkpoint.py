import torch
from safetensors.torch import save_file
import os

# --- Step 1: CONFIGURE YOUR PATHS HERE ---

# The full path to your original, incompatible checkpoint
original_checkpoint_path = "/home/cvlab20/project/hyunbin/Self-Forcing/checkpoints/self_forcing_dmd.pt"

# The desired path for your new, converted checkpoint
# Let's save it in the same directory with a new name.
#output_folder = os.path.dirname(original_checkpoint_path)
new_checkpoint_path = os.path.join("converted_checkpoint.safetensors")


# --- Step 2: THE SCRIPT WILL DO THE REST ---

print(f"Loading original checkpoint from: {original_checkpoint_path}")
try:
    # Load the original data structure
    data = torch.load(original_checkpoint_path, map_location="cpu")
    
    # Extract the nested state dictionary
    if 'generator_ema' in data:
        print("Found 'generator_ema' key. Extracting the nested state dictionary...")
        state_dict = data['generator_ema']
        
        # Save the extracted state_dict to a new file
        print(f"Saving converted checkpoint to: {new_checkpoint_path}")
        save_file(state_dict, new_checkpoint_path)
        
        print("\n✅ Conversion successful!")
    else:
        print("\nError: Could not find the 'generator_ema' key in the checkpoint.")

except Exception as e:
    print(f"\nAn error occurred: {e}")