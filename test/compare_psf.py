# plot heatmaps of cross entropies between PSFs

# %%
import nd2
import numpy as np
from pathlib import Path
from skimage import io,util,filters,restoration
from deconvolve_fft import deconvolve

# %%
names = [
    "zStack_PX-VO_FOV-0_camera-blues.nd2",
    "zStack_PX-VO-ER_FOV-8_camera-bluegreen.nd2",
]

for name in names:
    path = Path(f"../organelle-recognize/images/2025-02-23_Mixed1ColorDiploids/raw/{name}")
    img = nd2.imread(str(path))
    print(name, img.shape)
    for c in range(img.shape[1]):
        io.imsave(
            f"data/validate/tiff/c-{c}_{path.stem}.tiff",
            util.img_as_uint(img[:,c,...])
        )
# %%
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

# %% with or without background subtraction
psf = io.imread("data/psf/psf-mask_FOV-1_FITC.tiff")
for path in Path("data/validate/tiff").glob("c-3*.tiff"):
    raw = io.imread(str(path))

    bkgd = np.empty_like(raw, dtype=int)
    for z in range(raw.shape[0]):
        bkgd[z] = filters.gaussian(raw[z], sigma=25, preserve_range=True)
    img = raw - bkgd
    # img[img<0] = 0  # does this make a difference?

    deconv_raw = deconvolve(raw, psf, epsilon=1E-3)
    if deconv_raw.max() > 65535:
        deconv_raw = deconv_raw / deconv_raw.max() * 65535
    deconv_raw = deconv_raw.astype(np.uint16)
    io.imsave(
        f"data/validate/fft_mask/{path.stem}_raw.tiff",
        util.img_as_uint(deconv_raw)
    )
    
    deconv_img = deconvolve(img, psf, epsilon=1E-3)
    deconv_img = deconv_img.astype(np.uint16)
    if deconv_img.max() > 65535:
        deconv_img = deconv_img / deconv_img.max() * 65535
    io.imsave(
        f"data/validate/fft_mask/{path.stem}_clean.tiff",
        util.img_as_uint(deconv_raw)
    )

# %% richardson-lucy with theoretical PSF
for c,camera in zip([1,2,3],["DAPI","CFP","FITC"]):
    psf = io.imread(f"data/psf/psf-BornWolf_{camera}.tif")
    for path in Path("data/validate/tiff").glob(f"c-{c}*.tiff"):
        raw = io.imread(str(path))
        deconvolved = restoration.richardson_lucy(raw, psf, num_iter=50, clip=False)
        if deconvolved.max() > 65535:
            deconvolved = deconvolved / deconvolved.max() *65535
        deconvolved = deconvolved.astype(np.uint16)
        io.imsave(
            f"data/validate/rl_theoretical/{path.stem}_n-50.tiff",
            util.img_as_uint(deconvolved)
        )

# %% richardson-lucy with captured PSF
for c,camera in zip([1,2,3],["DAPI","CFP","FITC"]):
    psf = io.imread(f"data/psf/psf-mask_FOV-1_{camera}.tif")
    for path in Path("data/validate/tiff").glob(f"c-{c}*.tiff"):
        raw = io.imread(str(path))
        deconvolved = restoration.richardson_lucy(raw, psf, num_iter=50, clip=False)
        if deconvolved.max() > 65535:
            deconvolved = deconvolved / deconvolved.max() *65535
        deconvolved = deconvolved.astype(np.uint16)
        io.imsave(
            f"data/validate/rl_theoretical/{path.stem}_n-50.tiff",
            util.img_as_uint(deconvolved)
        )