import safetensors.torch
from collections import OrderedDict
import sys

# -----------------------------------------------------------
# Define your file paths
model_file_path = "/home/cvlab20/project/jinhyuk/DiffSynth-Studio/examples/wanvideo/model_training/wan_dit_from_self_forcing_CORRECTED.safetensors"
log_file_path = "self_forcing.log"  # The log file that will be created
# -----------------------------------------------------------

print(f"Loading tensors from: {model_file_path}")
# Load the tensors from the file
tensors = safetensors.torch.load_file(model_file_path, device="cpu")

# Sort the tensors by name for easier viewing
sorted_tensors = OrderedDict(sorted(tensors.items()))

# Open the log file in write mode ('w')
# The 'with' statement ensures the file is properly closed
with open(log_file_path, 'w') as log_file:
    print(f"Inspecting contents of: {model_file_path}\n", file=log_file)

    # Print the name, shape, and data type of each tensor
    for name, tensor in sorted_tensors.items():
        print(f"- Layer: {name}", file=log_file)
        print(f"  Shape: {tensor.shape}", file=log_file)
        print(f"  Type: {tensor.dtype}", file=log_file)
        print("-" * 20, file=log_file)

    print(f"\n✅ Found {len(sorted_tensors)} total tensors in the file.", file=log_file)

print(f"✅ Log file successfully saved to: {log_file_path}")