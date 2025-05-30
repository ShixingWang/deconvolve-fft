# %% [markdown] 
# We have the coordinates of the center of the beads in 2D (from `find_beads.py`)
# Now we need to crop the point spread function (PSF) around each coordinate.
# The challenge is that 
# 1. We set a constant z depth to find maxima, which may not be the true maxima.
# 2. We do not know the depth, height, and width of each psf.
# 3. Especially when they could be close to each other and appear in other PSFs
# Plan:
# 1. Use voronoi plot to partition the image
# 2. Find coords of maxima in each masked image.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import spatial,ndimage
from skimage import io,util,measure,draw

# %% Look at the intensity along z direction
filepath = Path("data/coordinates/FOV-1_DAPI.txt")
coordinates = np.loadtxt(str(filepath)).astype(int)
# %%
image = io.imread(f"data/clean/clean_{filepath.stem}.tiff")
# %%
for coord in coordinates:
    data = image[:,*[int(c) for c in coord]]
    plt.figure()
    plt.plot(data,label=f"{coord}, max at {int(np.argmax(data))}")
    plt.legend()
    plt.show()
    plt.close()
    # print(f"{coord},\t max at {np.argmax(data)}")

# %%
voronoi = spatial.Voronoi(coordinates)
# %%
markers = np.zeros((2044,2048),dtype=int)
for c,coord in enumerate(coordinates):
    markers[tuple(coord)] = c+1

distances, (y_idx, x_idx) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
label_image = markers[y_idx, x_idx]
# %%
io.imsave(
    "data/voronoi_FOV-1_DAPI.tiff",
    util.img_as_uint(label_image)
)

# %%
regions = np.empty_like(image,dtype=int)
for z in range(regions.shape[0]):
    regions[z] = label_image
centroids = measure.regionprops_table(
    label_image=regions,
    intensity_image=image,
    properties=('label',"centroid_weighted")
)
plt.hist(centroids["centroid_weighted-0"],bins=20,range=[0,50])
# This shows that direct weighted mean of the PSF does not give good estimate of center.

# %%
weighted = np.empty_like(image,dtype=int)
for e,entry in pd.DataFrame(centroids).iterrows():
    zz = entry["centroid_weighted-0"]
    if not 0 <= zz < regions.shape[0]:
        weighted[:,regions[regions==entry["label"]]] = entry["label"]
        continue
    rr,cc = draw.disk((entry["centroid_weighted-1"],entry["centroid_weighted-2"]),5,shape=(2044,2048))
    weighted[int(zz),rr,cc] = entry['label']
io.imsave(
    "data/weighted_FOV-1_DAPI.tiff",
    util.img_as_uint(weighted)
)
    

# %% [markdown] choice of psf window
# - DAPI:  20*20
# - FITC:  70 by 70
# - YFP:   20 by 20 
# - TRITC: 70 by 70
#  It's more of a signal-to-noise issue