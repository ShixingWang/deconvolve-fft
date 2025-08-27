# %% [markdown]
# This script basically shows that the PSF does not have to be symmetrical 
# around the center of the image. 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,fft
from skimage import filters,io,draw,util,metrics
from deconvolve_fft import deconvolve,calculate_pad2align

# %% VERIFY: the fftshift function according to ChatGPT
#    RESULT: ChatGPT is talking about something else.
obj = np.zeros((64,64))
obj[32,32] = 1

# PSF: small Gaussian, centered in array
x = np.arange(-8,8)
X,Y = np.meshgrid(x,x)
psf = np.exp(-(X**2+Y**2)/10)

# Case 1: original psf
blur1 = signal.convolve(obj,psf,method="fft",mode="same")

# Case 2: shifted psf
psf_shift = np.fft.fftshift(psf)
blur2 = signal.convolve(obj, psf_shift, method="fft", mode="same")

print("argmax of obj:", np.unravel_index(np.argmax(obj), obj.shape))
print("argmax of blur1 (psf centered):", np.unravel_index(np.argmax(blur1), blur1.shape))
print("argmax of blur2 (psf shifted):", np.unravel_index(np.argmax(blur2), blur2.shape))

plt.subplot(131); plt.imshow(obj); plt.title("Ground Truth")
plt.subplot(132); plt.imshow(blur1); plt.title("No shift (off-centered)")
plt.subplot(133); plt.imshow(blur2); plt.title("With fftshift (centered)")
plt.show()

# %% prepare data
obj = np.zeros((36,512,512),dtype=int)

for z in range(5):
    a = 120 - 5*z
    b =  72 - 3*z
    rr0,cc0 = draw.ellipse(256,256,120,72,rotation=0.5)
    obj[15-z,rr0,cc0] = 128 - z*8
    obj[15+z,rr0,cc0] = 128 - z*8

rr1,cc1 = draw.ellipse_perimeter(128,384,120,60,orientation=-0.5)
obj[20,rr1,cc1] = 96

rr2,cc2 = draw.ellipse_perimeter(384,128,120,60,orientation=1.0)
obj[10,rr2,cc2] = 192
# %%
io.imsave(
    "data/concept/obj.tiff",
    util.img_as_ubyte(obj)
)

# %%
psf = np.zeros((25,25,25))
psf[12,12,12] = 1
psf[18,16,14] = 0.5
psf = filters.gaussian(psf,sigma=1)
psf = psf/psf.sum()
# %%
io.imsave(
    "data/concept/psf.tiff",
    util.img_as_float32(psf)
)

# %%
img = signal.convolve(obj,psf,method="fft",mode="same")
img = img + np.random.poisson(lam=0.05*img.max(), size=img.shape)
img = img.astype(int) 
# %%
io.imsave(
    "data/concept/img.tiff",
    util.img_as_ubyte(img)
)


# %% 
def _new_deconvolve(image, psf, epsilon=0.):
    fft_img = fft.rfftn(image)

    shift_psf = fft.fftshift(psf)
    fft_psf = fft.rfftn(shift_psf)
    conj_fft = np.conjugate(fft_psf)

    fft_obj = fft_img * conj_fft/(fft_psf * conj_fft + epsilon)
    return fft.irfftn(fft_obj)

def new_deconvolve(image,psf,epsilon=0.):
    """Deconvolve the image with the PSF which do not have to be of the same size (will be padded by this function)."""
    pad1,pad2 = calculate_pad2align([image.shape, psf.shape])
    image = np.pad(image, pad_width=pad1)
    psf   = np.pad(psf,   pad_width=pad2)
    deconvolved = _new_deconvolve(image,psf,epsilon)
    deconvolved = deconvolved[tuple([slice(p[0],p[0]+s) for p,s in zip(pad1,image.shape)])]

    # mean_old = image.mean()
    # mean_new = deconvolved.mean()
    return deconvolved # - mean_new + mean_old

deconv0 = deconvolve(img,psf,epsilon=1/10)
deconv1 = new_deconvolve(img,psf,epsilon=1/10)
print("2 Deconvolution Methods are equal:", np.allclose(deconv0,deconv1))

# %% deconvolve (TODO: might need very small gaussian smoothing)

mean_obj = obj.mean()
mean_img = img.mean()
target = obj.astype(np.ubyte)

params = [10,100,1000,10000,1000000]
rmse = []
psnr = []
ssim = []
for k in params:
    deconvolved = deconvolve(img, psf, epsilon=1/k)
    deconvolved = filters.gaussian(deconvolved, sigma=0.5, preserve_range=True)
    mean_deconv = deconvolved[deconvolved>0].mean()
    predict = (deconvolved - mean_deconv + mean_img)
    predict[predict < 0] = 0
    predict = predict.astype(np.ubyte)
    rmse.append(metrics.mean_squared_error(target, predict))
    psnr.append(metrics.peak_signal_noise_ratio(target, predict))
    ssim.append(metrics.structural_similarity(target, predict))
    io.imsave(
        f"data/concept/prediction_1-{k}.tiff",
        util.img_as_ubyte(predict)
    )
benchmark = (img)
benchmark[benchmark < 0] = 0
benchmark = benchmark.astype(np.ubyte)
print("RMSE:",metrics.mean_squared_error(target, benchmark),rmse)
print("PSNR:",metrics.peak_signal_noise_ratio(target, benchmark),psnr)
print("SSIM:",metrics.structural_similarity(target, benchmark),ssim)

# When noise exists, there is a sweet spot for the choice of epsilon.
# if epsilon is too small, the noise will be amplified.
# if epsilon is too large, the deconvolution will not be applied to the image.
# Probelm: quantified metrics shows a different sweet spot than visual inspection.
# It turns out 
# - necessary to compensate the mean of the deconvolved images
# - not yet clear how helpful (if at all) the gaussian smoothing is.


# %%
