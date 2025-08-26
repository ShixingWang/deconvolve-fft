# plot heatmaps of cross entropies between PSFs

# %%
import nd2
import numpy as np
from pathlib import Path
from skimage import io,util,filters,restoration
from deconvolve_fft import deconvolve

# %% files to test
names = [
    # "zStack_PX-VO_FOV-0_camera-blues",
    "zStack_PX-VO-ER_FOV-8_camera-bluegreen",
]

# %% save each channel as z-stack tiff image
for name in names:
    path = Path(f"../organelle-recognize/images/2025-02-23_Mixed1ColorDiploids/raw/{name}.nd2")
    img = nd2.imread(str(path))
    print(name, img.shape)
    for c in range(img.shape[1]):
        io.imsave(
            f"data/validate/tiff/c-{c}_{path.stem}.tiff",
            util.img_as_uint(img[:,c,...])
        )
# %% save the background as blurred images with different gaussian sigmas
for path in Path("data/validate/tiff").glob("c-2*.tiff"):
    raw = io.imread(str(path))
    for sigma in [10,25,50,100]:
        img = np.empty_like(raw, dtype=int)
        for z in range(raw.shape[0]):
            img[z] = filters.gaussian(raw[z], sigma=sigma, preserve_range=True)
        io.imsave(
            f"data/validate/bkgd/{path.stem}_sigma-{sigma}.tiff",
            util.img_as_uint(img)
        )
# seems like sigma=25 is a good choice
# [2025-08-25] now I wonder how good this observation is without seeing the subtracted images.
#              but anyway sigma=25 is the choice when extracting PSFs above.

# %% with ~~or without~~ background subtraction
psf = io.imread("data/psf/from_raw/psf-raw_FOV-2_FITC.tiff")
psf = psf / psf.sum()
# psf = io.imread("data/psf/psf-BornWolf_CFP.tif")
# for path in Path("data/validate/tiff").glob("c-3*.tiff"):
for name in names:
    path = Path(f"data/validate/tiff/c-2_{name}.tiff")
    raw = io.imread(str(path))

    bkgd = np.empty_like(raw, dtype=int)
    for z in range(raw.shape[0]):
        bkgd[z] = filters.gaussian(raw[z], sigma=25, preserve_range=True)
    img = raw - bkgd
    # img[img<0] = 0  # does this make a difference?
    mean_img = img.mean()

    for m in range(1,12):
        deconv_img = deconvolve(img, psf, epsilon=1/10**m)
        # deconv_img = filters.gaussian(deconv_img, sigma=0.5, preserve_range=True)
        mean_deconv = deconv_img.mean()
        deconv_img = deconv_img - mean_deconv + mean_img
        # deconv_img[deconv_img < 0] = 0
        # deconv_img[deconv_img > 65535] = 65535
        deconv_img = deconv_img.astype(float)
        io.imsave(
            f"data/validate/fft/raw_{path.stem}_epsilon-1E-{m}.tiff",
            util.img_as_float32(deconv_img)
        )
        # epsilon = 1E-3 gives nice results on CFP with FITC PSF


# %% [markdown] 
# ## Richardson-Lucy 
# 
# This method results in uneven background across z-slices, even with background subtraction.

# %% richardson-lucy with theoretical PSF
for c,camera in zip([1,2,3],["DAPI","CFP","FITC"]):
    psf = io.imread(f"data/psf/psf-BornWolf_{camera}.tif")
    psf = psf / psf.sum()
    # for path in Path("data/validate/tiff").glob(f"c-{c}*.tiff"):
    for name in names:
        path = Path(f"data/validate/tiff/c-{c}_{name}.tiff")
        raw = io.imread(str(path))
        bkgd = np.empty_like(raw, dtype=int)
        for z in range(raw.shape[0]):
            bkgd[z] = filters.gaussian(raw[z], sigma=25, preserve_range=True)
        img = raw - bkgd
        for n in range(1,51,10):
            deconvolved = restoration.richardson_lucy(img, psf, num_iter=n, clip=False)
            deconvolved[deconvolved > 65535] = 65535
            deconvolved = deconvolved.astype(np.uint16)
            io.imsave(
                f"data/validate/rl/clean-theoretical_{path.stem}_n-{n}.tiff",
                util.img_as_uint(deconvolved)
            )
        # nearly 4 min for n=30, around 7 min for n=50

# %% richardson-lucy with captured PSF
for c,camera in zip([3],["FITC"]):
    psf = io.imread(f"data/psf/psf-max_FOV-1_{camera}.tiff")
    # for path in Path("data/validate/tiff").glob(f"c-{c}*.tiff"):
    for name in names:
        path = Path(f"data/validate/tiff/c-{c}_{name}.tiff")
        raw = io.imread(str(path))
        bkgd = np.empty_like(raw, dtype=int)
        for z in range(raw.shape[0]):
            bkgd[z] = filters.gaussian(raw[z], sigma=25, preserve_range=True)
        img = raw - bkgd
        for n in range(1,51,10):
            deconvolved = restoration.richardson_lucy(raw, psf, num_iter=n, clip=False)
            deconvolved[deconvolved > 65535] = 65535
            deconvolved = deconvolved.astype(np.uint16)
            io.imsave(
                f"data/validate/rl/clean-max_{path.stem}_n-{n}.tiff",
                util.img_as_uint(deconvolved)
            )
# %%
