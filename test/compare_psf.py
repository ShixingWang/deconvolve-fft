# plot heatmaps of cross entropies between PSFs

# %%
import nd2
import numpy as np
from pathlib import Path
from skimage import io,util,filters,restoration

# %%
names = [
    "zStack_PX-VO_FOV-0_camera-blues.nd2",
    "zStack_PX-VO-ER_FOV-8_camera-bluegreen.nd2",
]

for name in names:
    path = Path(f"../organelle-recognize/images/2025-02-23_Mixed1ColorDiploids/raw/{name}")
    img = nd2.imread(str(path))
    print(name, img.shape)
    for c in range(img.shape[1]):
        io.imsave(
            f"data/validate/tiff/c-{c}_{path.stem}.tiff",
            util.img_as_uint(img[:,c,...])
        )
# %%
for path in Path("data/validate/tiff").glob("c-2*.tiff"):
    raw = io.imread(str(path))
    for sigma in [10,25,50,100]:
        img = np.empty_like(raw, dtype=int)
        for z in range(raw.shape[0]):
            img[z] = filters.gaussian(raw[z], sigma=sigma, preserve_range=True)
        io.imsave(
            f"data/validate/bkgd/{path.stem}_sigma-{sigma}.tiff",
            util.img_as_uint(img)
        )
