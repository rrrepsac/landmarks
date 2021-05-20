#%%
from mutils.unets import LandmarksLoss
import numpy as np
lmLoss = LandmarksLoss(mode='round_5gauss', sigma=1.)
shape = (224,224)
bell = np.zeros(shape)
bell = lmLoss.bell[32:-32, 32:-32].numpy()
import matplotlib.pyplot as plt
plt.imshow(bell)
plt.show()
#%%
import mutils.PIL_test
import mutils
import importlib
import torch
import numpy as np
#%%
def ga(r, sig=1., e=0.):
    return np.exp(-(r - e)**2 / (2 * sig**2)) / (sig * np.sqrt(2*np.pi))

x = np.linspace(-3, 3, 21)
y = np.zeros_like(x)
print(x)
for k in range(5):
    s = 2*k + 1
    y += 2/5*np.pi * ga(x, s) * s**2
    print(*[f'{y_:.1e}' for y_ in y])
y *= 0
y = mutils.unets.bell_5gauss(x)
print(y)
#%%
import importlib
import mutils
import mutils.PIL_test

importlib.reload(mutils.PIL_test)
mutils.PIL_test.test_unet(resize=8, landmarks_number=2, batch_size=8,
max_affine_deg=50., distortion_scale=0.6)
assert False
# PIL_test.test_cnn(16, 3, 1, epochs=2, shuffle=False,
                #   max_affine_deg=0, distortion_scale=0.99)

# a = torch.Tensor([1,2,np.nan,np.inf])
# print(a.isnan())
# a[a.isnan()]=-1
# print(a)
from mutils.PIL_test import FaceLandmarks
from torch.utils.data import DataLoader
from mutils.unets import *
from pathlib import Path

if __name__ == '__main__':    
    print('__main__')
    ufl = UFLandmarks(8)
    x = torch.randn(1,3,224,224)
    yt = torch.randn(1,8,224,224)
    yp = ufl(x)
    loss = DiceLoss()(yt, yp)
    print(loss.item())

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import pi, sqrt, exp
    from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage,\
        RandomAffine, RandomPerspective

    s2p = sqrt(2 * pi)
    a = 1.
    sig = 1.
    x = np.linspace(-5, 5, 1000)
    y = exp(-((x - a) / (2 * sig)) ** 2) / s2p
    # plt.scatter(x, y)
    # plt.show()

    landmarks_number=3
    batch_size=1
    epochs=5
    shuffle=True
    max_affine_deg=120
    distortion_scale=0.8
    resize = 2
    new_size = 224

    ds = FaceLandmarks(Path('./face_landmarks.zip'),
                    landmarks_number, mode='tensor', resize=resize,
                    distortion_scale=distortion_scale, max_affine_degree=max_affine_deg, new_size=new_size)
    for batch in DataLoader(ds):
        pass
        img = ToPILImage()(batch[0].squeeze(0))
        ax = plt.subplot(121)
        ax.imshow(img)
        gauss_lm = make_gauss_landmarks(img.size, batch[1][0], sigma=4.)
        ax = plt.subplot(122)
        ax.imshow(ToPILImage()(gauss_lm.squeeze(0)))
        # for mark_num in range(gauss_lm.shape[0]):
            # ax = plt.subplot(1, 4, mark_num + 2)
            # ax.imshow(ToPILImage()(gauss_lm[mark_num]))
        plt.show()

        break

# %%
