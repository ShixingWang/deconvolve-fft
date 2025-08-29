# PREVIOUS: nd2tiff.py
# %%
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import neighbors
from skimage import util,io,restoration,morphology,filters,feature

# folders = ["bkgd","clean"]
# for f in folders:
#     if not Path(f"data/dev/{f}").is_dir():
#         Path.mkdir(f"data/dev/{f}")

# %% background subtraction: gaussian of sigma=25
%%time
# 2 images of (47,2044,2048), sigma=25:
# CPU times: total: 1min 26s
# Wall time: 1min 40s

# 12 images of (47,2044,2048), sigma=25:
# CPU times: total: 8min
# Wall time: 9min 13s

for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    image = io.imread(str(path))

    background = np.empty_like(image,dtype=int)
    for z in range(image.shape[0]):
        background[z] = filters.gaussian(image[z],sigma=25,preserve_range=True).astype(int)
    
    clean = image - background
    abslu = np.copy(clean)
    abslu[ abslu < 0 ] =  - abslu[ abslu < 0 ]

    io.imsave(
        f"data/dev/bkgd/{path.name}",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{path.name}",
        util.img_as_uint(clean)
    )
    io.imsave(
        f"data/dev/clean/abs_{path.name}",
        util.img_as_uint(abslu)
    )

# %% [markdown]
# % % time
# 2 images of (47,2044,2048), size0=25, size1=30:
# CPU times: total: 34.5 s
# Wall time: 44.5 s
# This method will generate a square bright ring around each bright center

# ```python
# for path in Path("data/dev/tiff").glob("FOV*FITC.tiff"):
#     image = io.imread(str(path))

#     average0 = np.empty_like(image,dtype=float)
#     average1 = np.empty_like(image,dtype=float)
#     for z in range(image.shape[0]):
#         average0[z] = ndi.uniform_filter(image[z],size=25)
#         average1[z] = ndi.uniform_filter(image[z],size=30)
#     background = ((average1 * 900 - average0 * 625) / (900 - 625)).astype(int)

#     clean = image - background
    
#     abslu = np.copy(clean)
#     abslu[abslu<0] = - abslu[abslu<0]

#     io.imsave(
#         f"data/dev/bkgd/diffmean/{path.name}",
#         util.img_as_uint(background)
#     )
#     io.imsave(
#         f"data/dev/clean/diffmean/{path.name}",
#         util.img_as_uint(clean)
#     )
#     io.imsave(
#         f"data/dev/clean/diffmean/abs_{path.name}",
#         util.img_as_uint(abslu)
#     )
# ```
# %%
# NEXT: ./locate_psf.py