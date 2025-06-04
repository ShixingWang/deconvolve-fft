# %% [markdown]
# The previous 2 scripts tried to first find the local maxima and then its neighboring PSFs.
# This script tries to segment the PSFs first, and find the centroids of them.

# %%
import numpy as np
from pathlib import Path
from skimage import io,util,filters,measure,morphology

# %%
image = io.imread("data/clean/FOV-1_DAPI.tiff")

# %% result too low
threshold = filters.threshold_otsu(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/otsu_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %% super slow
threshold = filters.threshold_li(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/li_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %%
threshold = filters.threshold_triangle(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/triangle_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %%
threshold = np.mean(image) + 1 * np.std(image)
binary = (image > threshold)
binary = morphology.binary_opening(binary, np.ones((3,3,3)))
io.imsave(
    "data/segmented/stdv_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %%
