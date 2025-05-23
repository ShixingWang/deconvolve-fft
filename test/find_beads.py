# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,util,filters,feature

# %% test on 1 image
filepath = "data/clean/clean_FOV-1_DAPI.tiff"
image = io.imread(str(filepath))

# 
# %%
coordinates = feature.peak_local_max(image,min_distance=20)
# %%
image[tuple((c) for c in coordinates)]
# %%
