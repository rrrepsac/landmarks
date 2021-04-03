from mutils import PIL_test
import importlib
import torch
import numpy as np
importlib.reload(PIL_test)
PIL_test.test_cnn(16, 3, 1, epochs=2, shuffle=False,
                  max_affine_deg=0, distortion_scale=0.99)

# a = torch.Tensor([1,2,np.nan,np.inf])
# print(a.isnan())
# a[a.isnan()]=-1
# print(a)

