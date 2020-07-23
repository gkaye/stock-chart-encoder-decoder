from util.data import yahoo
import torch

f = torch.zeros(10)
z = torch.tensor([[2.,2.,3.],[2.,2.,3.],[2.,2.,3.]])
print(z)

print(f)

yahoo.normalize_tensor(z)
print(z)


print("GREG")
