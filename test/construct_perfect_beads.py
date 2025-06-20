# %%
import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io,util,filters,morphology,measure

project_name = "2025-05-13_microspheresOnPetriDish"
channels = ["DAPI","FITC","YFP","TRITC"]

# %%
for fov in (1,2):
    beads2d = nd2.imread(f"data/raw/{project_name}/Frozen_FindObject_{fov}.nd2")
    threshold = filters.threshold_niblack(beads2d,window_size=151,k=-3)
    segmented = (beads2d > threshold)
    segmented = morphology.binary_opening(segmented)
    io.imsave(
        f"data/beads2D/FOV-{fov}.tiff",
        util.img_as_ubyte(segmented)
    )

# %%
for fov in (1,2):
    bw2d = io.imread(f"data/beads2d/FOV-{fov}.tiff")
    label_image = measure.label(bw2d)
    # prop_table = pd.DataFrame(measure.regionprops_table(
    #     label_image,
    #     properties=("label","area")
    # ))
    mask_exclude = np.zeros_like(label_image, dtype=bool)
    mask_include = np.zeros_like(label_image, dtype=int)
    for prop in measure.regionprops(label_image):
        if prop.area > 150:
            mask_exclude[label_image==prop.label] = True
            continue
        mask_include[label_image==prop.label] = prop.label
    io.imsave(
        f"data/beads2d/FOV-{fov}_exclude.tiff",
        util.img_as_ubyte(mask_exclude)
    )
    io.imsave(
        f"data/beads2d/FOV-{fov}_include.tiff",
        util.img_as_uint(mask_include)
    )

# %% [markdown]
# 
# 2 ways of finding the centroids:
# 1. Using the     raw image, let the (now positive) min pixel value in each Z to be zero
# 2. Using the cleaned image, let the (now negative) min pixel value of  all Z to be zero
# 
# after finding the centroids, we assign the sum of the masked pixels 
# to the 8 pixels around the centroid (decimal coordinates), 
# with weights that reflect the distance from the centroid to the pixel index.

# %%
for fov in (1,2):
    mask2d = io.imread(f"data/beads2d/FOV-{fov}_include.tiff")
    for c in channels:
        intensities = io.imread(f"data/clean/FOV-{fov}_{c}.tiff")
        intensities = intensities - intensities.min()
        mask3d = np.stack([mask2d for _ in range(intensities.shape[0])], axis=0)
        for prop in measure.regionprops(label_image=mask3d,intensity_image=intensities):
            continue
