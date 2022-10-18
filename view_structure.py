from turtle import shape
import torch
import numpy as np
checkpoint = torch.load("/DISK1/home/jclou5/Project/ViT/ViT-pytorch/checkpoint/vit_b_16-c867db91.pth")
checkpoint2 = np.load("/DISK1/home/jclou5/Project/ViT/ViT-pytorch/checkpoint/ViT-B_16.npz")
# print(checkpoint.keys())
# lst = checkpoint2.files
# for item in lst:
#     print(checkpoint2[item])
# print(checkpoint2.files)
print(type(checkpoint2))
# over = torch.from_numpy(checkpoint2)
print(checkpoint2['embedding/kernel'].shape)
