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
    "data/test/obj.tiff",
    util.img_as_ubyte(obj)
)

# %%
psf = np.zeros((25,25,25))
psf[12,12,12] = 1
psf = filters.gaussian(psf,sigma=5)

# %%
io.imsave(
    "data/test/psf.tiff",
    util.img_as_float32(psf)
)

io.imshow(psf[12])

# %%
img = signal.convolve(obj,psf,method="fft") 
# %%
io.imsave(
    "data/test/decoy_conv.tiff",
    util.img_as_float32(img)
)


# %% validate fft in python
gauss1d = np.zeros(25)
gauss1d[12] = 1
gauss1d = filters.gaussian(gauss1d,sigma=3)
fft_gauss1d = fft.fft(gauss1d)
ifft_gauss1d = fft.ifft(fft_gauss1d)
plt.plot(gauss1d,label="original")
plt.plot(np.real(ifft_gauss1d),label="recovered")
plt.legend()

# %%
gauss3d = np.zeros((25,25,25))
gauss3d[12,12,12] = 1
gauss3d = filters.gaussian(gauss3d,sigma=3)
plt.imshow(gauss3d[12])

# %%
fft_gauss3d = fft.fftn(gauss3d)
ifft_gauss3d = fft.ifftn(fft_gauss3d)

# %%
plt.imshow(np.real_if_close(ifft_gauss3d[12]))
# %%
plt.plot(gauss3d[12,12],label="original")
plt.plot(ifft_gauss3d[12,12],label="recovered")
plt.legend()

# %% perform deconvolution
fft_img = fft.rfftn(img)

pad_psf_z0,pad_psf_r0,pad_psf_c0 = (np.array(img.shape) - np.array(psf.shape))//2
pad_psf_z1,pad_psf_r1,pad_psf_c1 =  np.array(img.shape) - np.array(psf.shape) - np.array((pad_psf_z0,pad_psf_r0,pad_psf_c0))
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
    "data/test/pred_r_shift_crop_recenter_newpadding.tiff",
    util.img_as_float32(pred_obj)
)
# %%
