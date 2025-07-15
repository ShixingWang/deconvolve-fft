# %% [markdown]
# What we have:
# - `data/2022-08-31_Beads`: 37*2048*2044
#     + dilution-0:   no single pixel dots
#     + dilution-20:  contains signals, but backgournd not uniform
#     + dilution-100_field-2: looks good enough, but only this one
# - `data/2022-11-08_beads_equal_pixel_size`: could not see in the blue channels
# - `data/2025-05-13_microspheresOnPetriDish`: dots are there, but background is a problem

# %%
import time
import nd2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,filters,util,restoration,morphology

# %% 
project_name = "2025-05-13_microspheresOnPetriDish"

def remove_bkgd(image):
    bkgd  = np.zeros_like(image,dtype=int)
    clean = np.zeros_like(image,dtype=int)
    for z in range(image.shape[0]):
        bkgd[z]  = filters.gaussian(image[z], sigma=25, preserve_range=True)
        clean[z] = image[z] - bkgd[z]
    return clean,bkgd

for filepath in Path(f"data/raw/{project_name}").glob("FOV*.nd2"):
    raw  = nd2.imread(str(filepath))

    clean,bkgd = remove_bkgd(raw)
    io.imsave(
        f"data/clean/{filepath.stem}.tiff",
        util.img_as_uint(clean)
    )
    io.imsave(
        f"data/bkgd/{filepath.stem}.tiff",
        util.img_as_uint(bkgd)
    )
# %% [markdown] ## Scratch Zone
# %% before finding the PSF centers, 
# we need to confirm the background is not uniform and fix it.

filepath = "data/raw/2025-05-13_microspheresOnPetriDish/FOV-1_DAPI.nd2"
raw = nd2.imread(filepath)
N = 200
# %% 
for z in range(raw.shape[0]):
    if not z==24: continue
    plt.figure()
    plt.title(f"Intensity along the columnn at different rows, at {z=}")
    for r in range(0,raw.shape[1],N):
        data = raw[z,r]
        # data = (raw[z,r] - raw[z,r].min()) / (raw[z,r].max() - raw[z,r].min())
        plt.plot(data,label=f"{z=}, {r=}")
        plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.title(f"Intensity along the rows at different columns, at {z=}")
    for c in range(0,raw.shape[2],N):
        data = raw[z,:,c]
        # data = (raw[z,c] - raw[z,c].min()) / (raw[z,c].max() - raw[z,c].min())
        plt.plot(data,label=f"{z=}, {c=}")
        plt.legend()
    plt.show()
    plt.close()

# Observation: 
# Absolute intensities vary from different zs, rows and colunms but they share similar trends.
# Nomalization between max and min seems enough to converge different rows and columns.
# This indicates large sigma gaussian filtering on each z could work.

# %%
plt.figure()
# plt.title("Intensity along z axis, at different locations on the sensor.")
for r in range(0,raw.shape[1],1000):
    for c in range(0,raw.shape[2],1000):
        plt.plot(raw[:,r,c],label=f"{r=}, {c=}")
plt.legend()
plt.show()
plt.close()

# %%
cleaned = np.zeros_like(raw,dtype=float)
for z in range(raw.shape[0]):
    bkgd = filters.gaussian(raw[z],sigma=10,preserve_range=True) 
    cleaned[z] = raw[z] - bkgd

# %%
io.imsave(
    f"data/test/{Path(filepath).stem}.tiff",
    util.img_as_float32(cleaned)
)

# %% [markdown] 
# ### Alternative Cleaning
# ```python
# cleaned = np.zeros_like(raw,dtype=float)
# for z in range(raw.shape[0]):
#     normed = (raw[z] - raw[z].min()) / (raw[z].max() - raw[z].min()) # this normalization is problematic (includes signals)
#     bkgd = filters.gaussian(normed,sigma=10) # might need to tune param of reserve_range
#     cleaned[z] = normed - bkgd
# ````
# another idea would be to segment bead signals first

# %%
plt.figure()
for r in range(0,raw.shape[1],N):
    data = cleaned[24,r]
    # data = (raw[z,r] - raw[z,r].min()) / (raw[z,r].max() - raw[z,r].min())
    plt.plot(data,label=f"{r=}")
    plt.legend()
plt.show()
plt.close()

plt.figure()
for c in range(0,raw.shape[2],N):
    data = cleaned[24,:,c]
    # data = (raw[z,c] - raw[z,c].min()) / (raw[z,c].max() - raw[z,c].min())
    plt.plot(data,label=f"{c=}")
    plt.legend()
plt.show()
plt.close()

plt.figure()
# plt.title("Intensity along z axis, at different locations on the sensor.")
for r in range(0,cleaned.shape[1],1000):
    for c in range(0,cleaned.shape[2],1000):
        plt.plot(cleaned[:,r,c],label=f"{r=}, {c=}")
plt.legend()
plt.show()
plt.close()


# %% How should we normalize the background-clean image?
# 1. Between the max and min of the whole image.
# 2. Between the max and min of each z (No, that way every z slice will have a 1.0)
# 3. Difference from mean/median in terms of ratio to std. (Negative intensities)

# %% [markdown] 
# Clean the image based on 3D information is not a good idea
# ```python
# clean3d = np.zeros_like(raw,dtype=float)
# bkgd3d = filters.gaussian(raw,sigma=(1,10,10),preserve_range=True) 
# clean3d = raw - bkgd3d
# 
# plt.figure()
# for r in range(0,raw.shape[1],N):
#     data = clean3d[0,r]
#     # data = (raw[z,r] - raw[z,r].min()) / (raw[z,r].max() - raw[z,r].min())
#     plt.plot(data,label=f"{r=}")
#     plt.legend()
# plt.show()
# plt.close()
# 
# plt.figure()
# for c in range(0,raw.shape[2],N):
#     data = clean3d[0,c]
#     # data = (raw[z,c] - raw[z,c].min()) / (raw[z,c].max() - raw[z,c].min())
#     plt.plot(data,label=f"{c=}")
#     plt.legend()
# plt.show()
# plt.close()
# ```
# %%
