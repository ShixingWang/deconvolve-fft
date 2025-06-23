# %% [markdown]
# The previous 2 scripts tried to first find the local maxima and then its neighboring PSFs.
# This script tries to segment the PSFs first, and find the centroids of them.

# %%
import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,filters,measure,morphology

# %%
project_name = "2025-05-13_microspheresOnPetriDish"

def threshold_stdv(image,window_size=251,k=-3,opener=morphology.disk(1)):
    binary = np.zeros_like(image,dtype=bool)
    for z in range(image.shape[0]):
        threshold = filters.threshold_niblack(image[z],window_size=window_size,k=k)
        binarized = (image[z] > threshold)
        binary[z] = morphology.binary_opening(binarized, opener)    
    return binary

for filepath in Path(f"data/raw/{project_name}").glob("FOV*.nd2"):
    img = nd2.imread(str(filepath))
    binary = threshold_stdv(img)
    io.imsave(
        f"data/segmented/{filepath.stem}.tiff",
        util.img_as_ubyte(binary)
    )

# %% test on 1 image
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
binary = morphology.binary_opening(
    binary, 
    # np.ones((3,3,3))，
    morphology.ball(1)
)
# %% save
io.imsave(
    "data/segmented/stdv_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# there could be false positives in the above thresholded image
# when the element of binary_opening is a ball(1), 
# those false positives are also ball(1)

