# PREVIOUS: ./clean_background.py
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,feature,morphology

# %% the background structure (mean, standard deviation) varies across z
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
    

# %% simply subtract the min of each slice is not enough
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
# for path in Path("data/dev/tiff").glob("FOV*.tiff"):
for path in Path("data/dev/clean").glob("FOV*.tiff"):
    image = io.imread(str(path))
    # subtracted = np.empty_like(image)
    # for z in range(10,image.shape[0]-10):
    #     min = image[z].min()
    #     subtracted[z] = image[z] - min
    # proj = np.max(subtracted,axis=0)
    proj = np.max(image,axis=0)
    io.imsave(
        f"data/dev/proj/{path.name}",
        util.img_as_uint(proj)
    )
# %%
thresh_rels = {
    "1-DAPI":  1/4,
    "1-YFP":   1/20,
}
for path in Path("data/dev/proj").glob("FOV*.tiff"):
    camera = path.stem.rpartition("_")[2]
    fov = path.stem.partition("FOV-")[2].partition("_")[0]

    if not f"{fov}-{camera}" in thresh_rels.keys():
        continue

    proj = io.imread(str(path))

    rel = thresh_rels[f"{fov}-{camera}"] if f"{fov}-{camera}" in thresh_rels.keys() else 1/5
    coords = feature.peak_local_max(proj,min_distance=10,threshold_rel=rel)
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
