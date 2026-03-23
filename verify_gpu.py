# Save as verify_gpu.py and run it
import torch

print("=== GPU Verification ===")
print(f"PyTorch version:     {torch.__version__}")
print(f"CUDA available:      {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:            {torch.cuda.get_device_name(0)}")
    print(f"CUDA version:        {torch.version.cuda}")
    print(f"GPU Memory:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quick tensor test on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"Matrix multiply on GPU: ✅ Shape {z.shape}")
else:
    print("⚠️  No GPU detected — training will use CPU")