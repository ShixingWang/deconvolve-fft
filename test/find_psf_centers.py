# %% [markdown]
# What we have:
# - `data/2022-08-31_Beads`: 37*2048*2044
#     + dilution-0:   no single pixel dots
#     + dilution-20:  contains signals, but backgournd not uniform
#     + dilution-100_field-2: looks good enough, but only this one
# - `data/2022-11-08_beads_equal_pixel_size`: could not see in the blue channels
# - `data/2025-05-13_microspheresOnPetriDish`: dots are there, but background is a problem

# %%
import nd2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io,feature,filters

# %% before finding the PSF centers, 
# we need to confirm the background is not uniform and fix it.

filepath = "data/raw/2025-05-13_microspheresOnPetriDish/FOV-1_DAPI.nd2"
raw = nd2.imread(filepath)
# %% 
N = 200
for z in range(raw.shape[0]):
    if not z==5: continue
    plt.figure()
    for r in range(0,raw.shape[1],N):
        data = raw[z,r]
        # data = (raw[z,r] - raw[z,r].min()) / (raw[z,r].max() - raw[z,r].min())
        plt.plot(data,label=f"{z=}, {r=}")
        plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    for c in range(0,raw.shape[2],N):
        data = raw[z,c]
        # data = (raw[z,c] - raw[z,c].min()) / (raw[z,c].max() - raw[z,c].min())
        plt.plot(data,label=f"{z=}, {c=}")
        plt.legend()
    plt.show()
    plt.close()

# Observation: 
# Absolute intensities vary from different rows and colunms but they share similar trends.
# Nomalization between max and min seems enough to converge different rows and columns.
# This indicates large sigma gaussian filtering on each z could work.

# %%
cleaned = np.zeros_like(raw,dtype=float)
for z in range(raw.shape[0]):
    normed = (raw[z] - raw[z].min()) / (raw[z].max() - raw[z].min()) # this normalization is problematic (includes signals)
    bkgd = filters.gaussian(normed,sigma=10) # might need to tune param of reserve_range
    cleaned[z] = normed - bkgd

    # therefore?
    bkgd = filters.gaussian(normed,sigma=10,preserve_range=True) 
    cleaned[z] = raw[z] - bkgd
