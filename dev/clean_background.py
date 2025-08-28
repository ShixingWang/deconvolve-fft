# %%
import numpy as np
from pathlib import Path
from skimage import util,io,restoration,morphology,filters

folders = ["bkgd","clean"]
for f in folders:
    if not Path(f"data/dev/{f}").is_dir():
        Path.mkdir(f"data/dev/{f}")

# %% [markdown]
# ## Pipeline
# 1. Flat field correction & background subtraction
# 2. Bead detection with max projection & peak finding
# 3. Bead cropping (normalize to max = 1 at this step)
# 4. PSF alignment (it suggests Fourier shift theorem)
# 5. Average (a new idea: use median ) & normalzie to sum = 1

# %% [markdown]
# ## 1. Flat field correction & background subtraction

# %% ChatGPT's suggestion: `skimage.restoration.rolling_ball`
%%time
# CPU times: total: 18min 8s
# Wall time: 19min 6s

method = "rolling_ball"

for f in folders:
    if not Path(f"data/dev/{f}/{method}").is_dir():
        Path.mkdir(f"data/dev/{f}/{method}")
        print(f"<clean_bkgd.{method}>: made new directory: `data/dev/{f}/{method}`")

for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    img = io.imread(str(path))

    background = np.empty_like(img,dtype=int)
    for z in range(img.shape[0]):
        background[z] = restoration.rolling_ball(
            filters.gaussian(img[z],sigma=0.5,preserve_range=True).astype(int),
            radius=10
        )
    clean = img - background
    
    io.imsave(
        f"data/dev/bkgd/{method}/{path.stem}.tiff",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{method}/{path.stem}.tiff",
        util.img_as_uint(clean)
    )
    print(f"<clean_bkgd.{method}>: cleaned image at `{path}`")

# %% Scikit-Image's warning: rolling_ball is slow; try top hat filter
%%time
# CPU times: total: 20min 20s
# Wall time: 21min 19s


method = "top_hat"

for f in folders:
    if not Path(f"data/dev/{f}/{method}").is_dir():
        Path.mkdir(f"data/dev/{f}/{method}")
        print(f"<clean_bkgd.{method}>: made new directory: `data/dev/{f}/{method}`")

for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    img = io.imread(str(path))
    
    footprint = morphology.disk(10)
    clean = np.empty_like(img,dtype=int)
    for z in range(img.shape[0]):
        clean[z] = morphology.white_tophat(img[z],footprint)
    background = img - clean

    io.imsave(
        f"data/dev/bkgd/{method}/{path.stem}.tiff",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{method}/{path.stem}.tiff",
        util.img_as_uint(clean)
    )
    print(f"<clean_bkgd.{method}>: cleaned image at `{path}`")

# %% F**k them all, what about just median filter with a large window
%%time
# CPU times: total: 1h 53min 52s
# Wall time: 1h 55min 13s
# Never mind, it is even slower.

method = "median"

for f in folders:
    if not Path(f"data/dev/{f}/{method}").is_dir():
        Path.mkdir(f"data/dev/{f}/{method}")
        print(f"<clean_bkgd.{method}>: made new directory: `data/dev/{f}/{method}`")

for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    img = io.imread(str(path))
    
    footprint = np.ones((20,20),dtype=int)
    background = np.empty_like(img,dtype=int)
    for z in range(img.shape[0]):
        background[z] = filters.median(img[z],footprint)
    clean = img - background

    io.imsave(
        f"data/dev/bkgd/{method}/{path.stem}.tiff",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{method}/{path.stem}.tiff",
        util.img_as_uint(clean)
    )
    print(f"<clean_bkgd.{method}>: cleaned image at `{path}`")


# %% Following previous idea: 
%%time
# 1. segment the signals from background with filters.niblack
# 2. gaussian smooth the background with a rather small sigma (more local)
# 3. fill the missing backgound on the signal pixels with restoration.inpaint_biharmonic

method = "inpaint"

for f in folders:
    if not Path(f"data/dev/{f}/{method}").is_dir():
        Path.mkdir(f"data/dev/{f}/{method}")
        print(f"<clean_bkgd.{method}>: made new directory: `data/dev/{f}/{method}`")

threshold = lambda arr: arr.mean() + arr.std() # * 1
for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    img = io.imread(str(path))
    
    mask = np.empty_like(img,dtype=bool)
    background = np.empty_like(img,dtype=int)
    for z in range(img.shape[0]):
        smooth = filters.gaussian(img[z],sigma=0.75,preserve_range=True)
        thresholds = filters.threshold_niblack(smooth,window_size=51,k=-1)
        mask_z = (smooth > thresholds) # mask[z]
        mask_z = morphology.binary_erosion(mask_z,footprint=np.ones((3,3)))
        mask[z] = morphology.binary_dilation(mask_z,footprint=np.ones((7,7)))
        background[z] = restoration.inpaint_biharmonic(smooth,mask[z])
    clean = img - background

    io.imsave(
        f"data/dev/bkgd/{method}/mask_{path.stem}.tiff",
        util.img_as_ubyte(mask)
    )
    io.imsave(
        f"data/dev/bkgd/{method}/{path.stem}.tiff",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{method}/{path.stem}.tiff",
        util.img_as_uint(clean)
    )
    print(f"<clean_bkgd.{method}>: cleaned image at `{path}`")



# %%
