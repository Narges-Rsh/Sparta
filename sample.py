import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (runtime):", torch.version.cuda)
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "No GPU found")
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
x = torch.randn(10, 3, 224, 224).cuda()
m = torch.nn.BatchNorm2d(3).cuda()
print(m(x).shape)