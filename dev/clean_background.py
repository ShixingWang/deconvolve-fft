# %%
import numpy as np
from pathlib import Path
from skimage import util,io,restoration,morphology,filters

folders = ["bkgd","clean"]
for f in folders:
    if not Path(f"data/dev/{f}").is_dir():
        Path.mkdir(f"data/dev/{f}")

# %%
%%time

for path in Path("data/dev/tiff").glob("FOV*.tiff"):
    image = io.imread(str(path))
    mask  = io.imread(f"data/dev/mask/{path.name}")
    mask = (mask==2)
    
    background = np.empty_like(image,dtype=int)
    for z in range(image.shape[0]):
        smooth = filters.gaussian(image[z],sigma=0.5,preserve_range=True)
        background[z] = restoration.inpaint_biharmonic(smooth,mask[z])
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
