# %%
import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io,util,filters,morphology,measure

project_name = "2025-05-13_microspheresOnPetriDish"
channels = ["DAPI","FITC","YFP","TRITC"]

# %%
for fov in (1,2):
    beads2d = nd2.imread(f"data/raw/{project_name}/Frozen_FindObject_{fov}.nd2")
    threshold = filters.threshold_niblack(beads2d,window_size=151,k=-3)
    segmented = (beads2d > threshold)
    segmented = morphology.binary_opening(segmented)
    io.imsave(
        f"data/beads2D/FOV-{fov}.tiff",
        util.img_as_ubyte(segmented)
    )

# %% [markdown]
# ```python
# for fov in (1,2):
#     bw2d = io.imread(f"data/beads2d/FOV-{fov}.tiff")
#     label_image = measure.label(bw2d)
#     prop_table = pd.DataFrame(measure.regionprops_table(
#         label_image,
#         properties=("label","area")
#     ))
#     plt.hist(prop_table["area"],bins=16)
# ``` 

# %%

