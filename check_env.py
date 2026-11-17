import torch, triton
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
print("triton", triton.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", p.name, "| sm", f"{p.major}.{p.minor}", "| vram", round(p.total_memory/2**30, 1), "GB")
