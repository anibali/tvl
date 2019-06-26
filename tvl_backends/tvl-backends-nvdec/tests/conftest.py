import torch

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))
