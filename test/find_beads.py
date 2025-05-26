# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,filters,feature,draw

# %%
def find_beads(image,z):

    return beads,coordinates

for filepath in Path("data/clean").glob("*FOV-1*.tiff"):
    img = io.imread(str(filepath))
    beads,coordinates = find_beads(img,z=12)


for filepath in Path("data/clean").glob("*FOV-1*.tiff"):
    img = io.imread(str(filepath))
    beads,coordinates = find_beads(img,z=20)
    



# %% test on 1 image
filepath = "data/clean/clean_FOV-1_DAPI.tiff"
image = io.imread(str(filepath))

# 
# %%
smoothed = filters.gaussian(image,sigma=1,preserve_range=True)
io.imsave(
    "data/smooth_FOV-1_DAPI.tiff",
    util.img_as_float32(smoothed)
)

# %%
coordinates = feature.peak_local_max(smoothed,min_distance=20,threshold_abs=10)
markers = np.zeros_like(smoothed,dtype=bool)
for c in coordinates:
    rr,cc = draw.disk(c[1:],5)
    markers[c[0],rr,cc] = True
io.imsave(
    "data/markers-3d_FOV-1_DAPI.tiff",
    util.img_as_ubyte(markers)
)

# %%
coordinates = feature.peak_local_max(smoothed[12],min_distance=20,threshold_abs=10)
markers = np.zeros_like(smoothed[12],dtype=bool)
for c in coordinates:
    rr,cc = draw.disk(c,5)
    markers[rr,cc] = True
io.imsave(
    "data/markers-2d_FOV-1_DAPI.tiff",
    util.img_as_ubyte(markers)
)
# %%
