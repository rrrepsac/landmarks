import numpy as np
import torch

a = torch.randn(2, 3,2)

print(a)

b = (torch.randn(5,1)*10 + 21).long()
W=10
bb = ([(x//W, x%W) for x in b])

print(bb)
bb = torch.tensor(bb)
print(bb)
print(b, bb)

