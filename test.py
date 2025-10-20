import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

# If CUDA is available, print more details
if cuda_available:
    # Print the CUDA version PyTorch was compiled with
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    
    # Print the name of the current GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch is running on CPU.")