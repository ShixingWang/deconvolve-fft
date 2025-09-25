# deconvolve bead images with the PSFs
# %%
import nd2
import numpy as np
from skimage import io,util
from scipy import fft
from deconvolve_fft import _deconvolve,deconvolve

# %% 
channels = (
            "DAPI",
            "FITC",
            "YFP",
            "TRITC"
            )
FoVs = (1,2)

# %% devonvolve the cleaned images (uniform backgrounds)
for c in channels:
    for f in FoVs:
        psf = io.imread(f"data/psf/psf-median_FOV-{f}_{c}.tiff")
        psf = psf / psf.sum()
        beads = io.imread(f"data/dev/clean/FOV-{f}_{c}.tiff")
        mean_beads = beads.mean()
        for k in range(1,9):
            deconvolved = deconvolve(beads,psf,epsilon=1/10**k)
            io.imsave(
                f"data/validate/fft/FOV-{f}_{c}_clean_epsilon-1E{k}.tiff",
                util.img_as_float32(deconvolved)
            )
# epsilon ↑: less deconvolved, more like blurry images
# epsilon ↓: more deconvolved, could give empty images
# epsilon = 1E-6 is good enough for FITC and TRITC.
# epsilon = 1E-3 is pretty good for DAPI and YFP.


# %%
