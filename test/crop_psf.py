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
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import spatial,ndimage
from skimage import io,util

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
print()