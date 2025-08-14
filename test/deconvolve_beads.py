# deconvolve bead images with the PSFs
# %%
import nd2
import numpy as np
from skimage import io,util
from scipy import fft
from deconvolve_fft import _deconvolve,deconvolve

# %% 
channels = ("DAPI","FITC","YFP","TRITC")
FoVs = (1,2)

# %% devonvolve the cleaned images (uniform backgrounds)
for c in ["FITC","TRITC"]:
    for f in FoVs:
        psf = io.imread(f"data/psf/psf-average_FOV-{f}_{c}.tiff")
        beads = io.imread(f"data/dev/clean/FOV-{f}_{c}.tiff")

        deconvolved = deconvolve(beads,psf,epsilon=1E-8).astype(int)
        if deconvolved.max() > 65535:
            print(f"Warning: FOV-{f}_{c} deconvolved image has values larger than 65535, clipping to 65535.")
            deconvolved = (65535 * deconvolved / deconvolved.max()).astype(int)

        io.imsave(
            f"data/dev/deconvolved/FOV-{f}_{c}_clean_epsilon-1E-8.tiff",
            util.img_as_uint(deconvolved)
        )
# epsilon ↑: less deconvolved, more like blurry images
# epsilon ↓: more deconvolved, could give empty images
# epsilon = 1E-6 is good enough for FITC and TRITC.
# epsilon = 1E-3 is pretty good for DAPI and YFP.

# %% devonvolve the raw images (non-uniform backgrounds)
import nd2

for c,ep in zip(channels,[1E-3,1E-6,1E-3,1E-6]):
    for f in FoVs:
        psf = io.imread(f"data/psf/psf-average_FOV-{f}_{c}.tiff")
        beads = nd2.imread(f"data/dev/raw/2025-05-13_microspheresOnPetriDish/FOV-{f}_{c}.nd2")

        deconvolved = deconvolve(beads,psf,epsilon=ep).astype(int)
        if deconvolved.max() > 65535:
            print(f"Warning: FOV-{f}_{c} deconvolved image has values larger than 65535, normalizing to 65535.")
            deconvolved = (65535 * deconvolved / deconvolved.max()).astype(int)
        io.imsave(
            f"data/dev/deconvolved/FOV-{f}_{c}_raw_epsilon-{str(ep).replace('.','-')}.tiff",
            util.img_as_uint(np.real_if_close(deconvolved))
        )

# %% looks nice directly on raw images, but we need some quantified metrics.

