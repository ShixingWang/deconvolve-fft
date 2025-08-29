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

# %%
for path in Path("data/dev/tiff").glob("FOV*FITC.tiff"):
    image = io.imread(str(path))
    plt.figure()
    for z in range(image.shape[0]):
        max = image[z].max()
        plt.hist(
            image[z].flatten(),
            np.array(range(0,max+2,1))-0.5,
            label=f"{z}",histtype='step'
        )
    plt.title(f"{path.stem}")
    # plt.legend()
    plt.show()
    

# %%
for path in Path("data/dev/tiff").glob("FOV*FITC.tiff"):
    image = io.imread(str(path))
    plt.figure()
    for z in range(10,image.shape[0]-10):
        min = image[z].min()
        data = image[z] - min
        max = data.max()
        plt.hist(
            data.flatten(),
            np.array(range(0,502,1))-0.5,
            label=f"{z}",histtype='step'
        )
    plt.title(f"{path.stem}")
    # plt.legend()
    plt.show()

# %%
for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    image = io.imread(str(path))
    subtracted = np.empty_like(image)
    for z in range(10,image.shape[0]-10):
        min = image[z].min()
        subtracted[z] = image[z] - min
    proj = np.max(subtracted,axis=0)
    io.imsave(
        f"data/dev/proj/{path.name}",
        util.img_as_uint(proj)
    )
# %%
thresh_rels = {
    "CFP":   2/3,
    "DAPI":  2/3,
    "FITC":  1/5,
    "YFP":   2/5,
    "TRITC": 1/5,
}
for path in Path("data/dev/proj").glob("FOV*.tiff"):
    camera = path.stem.rpartition("_")[2]
    proj = io.imread(str(path))

    coords = feature.peak_local_max(proj,min_distance=10,threshold_rel=thresh_rels[camera])
    np.savetxt(
        f"data/dev/pins/{path.stem}.txt",
        coords, fmt='%d'
    )

    pins = np.zeros_like(proj,dtype=bool)
    for r,c in coords:
        pins[r,c] = True
    pins = morphology.binary_dilation(pins,morphology.disk(5))
    io.imsave(
        f"data/dev/pins/{path.name}",
        util.img_as_ubyte(pins)
    )

# %%
for path in Path("data/dev/tiff").glob("FOV*FITC.tiff"):
    image = io.imread(str(path))
    holes = np.copy(image).astype(float)

    mask = np.zeros_like(image[0],dtype=bool)
    coords = np.loadtxt(f"data/dev/pins/{path.stem}.txt")
    for r,c in coords:
        mask[int(r),int(c)] = True
    mask = morphology.binary_dilation(mask,morphology.disk(40))
    holes[:,mask] = np.nan

    background = np.empty_like(image,dtype=float)
    for z in range(image.shape[0]):
        background[z] = ndi.generic_filter(
            holes[z], np.nanmean, 
            footprint=np.ones((100,100), dtype=int)
        )
    background = background.astype(int)

    clean = image - background

    io.imsave(
        f"data/dev/bkgd/{path.name}",
        util.img_as_uint(background)
    )
    io.imsave(
        f"data/dev/clean/{path.name}",
        util.img_as_uint(clean)
    )


# %%
