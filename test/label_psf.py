# original images exported by ilastik (1,z,y,x) cannot be opened by skimage
# the problem is solved by loading and saving by ilastik

import tifffile
import numpy as np
from pathlib import Path
from skimage import io,util,measure

for filepath in Path("data/probs/").glob("*.tiff"):
    probs = tifffile.imread(str(filepath))
    binary = (probs > 0.1)
    masks = measure.label(binary)

    formatter = util.img_as_ubyte if masks.max() < 256 else util.img_as_uint
    io.imsave(
        f"data/masks/{filepath.stem}.tiff",
        formatter(masks)
    )

# Then the masks will be manually checked in Fiji 
# and obviouly non-bead masks will be removed.