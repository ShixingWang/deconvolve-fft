# %% [markdown]
# This script basically shows that the PSF does not have to be symmetrical 
# around the center of the image. 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,fft
from skimage import filters,io,draw,util

# %% prepdra data
obj = np.zeros((36,512,512))

rr0,cc0 = draw.ellipse(256,256,120,60,rotation=0.5)
obj[15,rr0,cc0] = 1

rr1,cc1 = draw.ellipse_perimeter(128,384,120,60,orientation=-0.5)
obj[20,rr1,cc1] = 1

rr2,cc2 = draw.ellipse_perimeter(384,128,120,60,orientation=1.0)
obj[10,rr2,cc2] = 1
# %%
io.imsave(
    "data/obj.tiff",
    util.img_as_ubyte(obj)
)

# %%
psf = np.zeros((25,25,25))
psf[12,12,12] = 1
psf[18,16,14] = 0.5
psf = filters.gaussian(psf,sigma=5)
# %%
io.imsave(
    "data/psf.tiff",
    util.img_as_float32(psf)
)

# %%
img = signal.convolve(obj,psf,method="fft") 
# %%
io.imsave(
    "data/decoy_conv.tiff",
    util.img_as_float32(img)
)



# %% perform deconvolution
fft_img = fft.rfftn(img)

pad_psf_z1,pad_psf_r1,pad_psf_c1 = (np.array(img.shape) - np.array(psf.shape))//2
pad_psf_z0,pad_psf_r0,pad_psf_c0 =  np.array(img.shape) - np.array(psf.shape) - np.array((pad_psf_z1,pad_psf_r1,pad_psf_c1))
padded_psf = np.pad(psf,((pad_psf_z0,pad_psf_z1),(pad_psf_r0,pad_psf_r1),(pad_psf_c0,pad_psf_c1)))

fft_psf = fft.rfftn(padded_psf)

alpha = 0.0
fft_obj = fft_img * np.conjugate(fft_psf)/(np.abs(fft_psf)**2+alpha)

# %% check result
pred_obj = fft.ifftshift(fft.irfftn(fft_obj))
pad_img_z0,pad_img_r0,pad_img_c0 = (np.array(img.shape) - np.array(obj.shape))//2
pad_img_z1,pad_img_r1,pad_img_c1 =  np.array(img.shape) - np.array(obj.shape) - np.array((pad_img_z0,pad_img_r0,pad_img_c0))
pred_obj = pred_obj[pad_img_z0:-pad_img_z1,pad_img_r0:-pad_img_r1,pad_img_c0:-pad_img_c1]

# %%
io.imshow(pred_obj[30])
io.imsave(
    "data/pred_asym.tiff",
    util.img_as_float32(pred_obj)
)
# %%
