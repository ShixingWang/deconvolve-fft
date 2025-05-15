import numpy as np
from scipy import signal,fft
from skimage import filters,io,draw,util

obj = np.zeros((36,512,512))
rr0,cc0 = draw.ellipse(256,256,120,60,rotation=0.5)
obj[15,rr0,cc0] = 1
rr1,cc1 = draw.ellipse_perimeter(128,384,120,60,orientation=-0.5)
obj[20,rr1,cc1] = 1
rr2,cc2 = draw.ellipse_perimeter(384,128,120,60,orientation=1.0)
obj[10,rr2,cc2] = 1

io.imshow(obj[15])
io.imshow(obj[10])
io.imshow(obj[20])

psf = np.zeros((25,25,25))
psf[12,12,12] = 1
psf = filters.gaussian(psf,sigma=5)

io.imshow(psf[12])

img = signal.convolve(obj,psf,method="fft") 
io.imsave(
    "data/decoy_conv.tiff",
    util.img_as_float32(img)
)

pad_img_z,pad_img_r,pad_img_c = np.array(psf.shape)//2
pad_psf_z,pad_psf_r,pad_psf_c = np.array(img.shape)//2

padded_img = np.pad(img,((pad_img_z,pad_img_z),(pad_img_r,pad_img_r),(pad_img_c,pad_img_c)))
padded_psf = np.pad(psf,((pad_psf_z,pad_psf_z),(pad_psf_r,pad_psf_r),(pad_psf_c,pad_psf_c)))


