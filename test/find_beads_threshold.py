# %% [markdown]
# The previous 2 scripts tried to first find the local maxima and then its neighboring PSFs.
# This script tries to segment the PSFs first, and find the centroids of them.

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,filters,measure,morphology

# %%
def threshold_stdv(image):
    threshold = np.mean(image) + 1 * np.std(image)
    binary = (image > threshold)
    binary = morphology.binary_opening(binary, morphology.ball(1))
    return binary

for filepath in Path("data/clean").glob("*.tiff"):
    img = io.imread(str(filepath))
    binary = threshold_stdv(img)
    io.imsave(
        f"data/segmented/{filepath.stem}.tiff",
        util.img_as_ubyte(binary)
    )

# %% test on 1 image
image = io.imread("data/clean/FOV-1_DAPI.tiff")

# %% result too low
threshold = filters.threshold_otsu(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/otsu_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %% super slow
threshold = filters.threshold_li(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/li_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %%
threshold = filters.threshold_triangle(image)
binary = (image > threshold)
io.imsave(
    "data/segmented/triangle_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# %%
threshold = np.mean(image) + 1 * np.std(image)
binary = (image > threshold)
binary = morphology.binary_opening(
    binary, 
    # np.ones((3,3,3))，
    morphology.ball(1)
)
# %% save
io.imsave(
    "data/segmented/stdv_FOV-1_DAPI.tiff",
    util.img_as_ubyte(binary)
)

# there could be false positives in the above thresholded image
# when the element of binary_opening is a ball(1), 
# those false positives are also ball(1)


# %% [markdown] This script helps find the size ranges of PSFs in each channel. 
# ```python
# filepath = Path("data/segmented/FOV-2_FITC.tiff")
# mask = io.imread(str(filepath))
# intensities = io.imread(f"data/clean/{filepath.stem}.tiff")
# label_image = measure.label(mask)
# props_table = pd.DataFrame(measure.regionprops_table(
#     label_image=label_image,
#     intensity_image=intensities,
#     properties=('label','area','centroid_weighted','bbox')
# ))
# selected_table = props_table[
#     props_table["area"].gt(20)
#   & props_table["area"].lt(5000)
# ]
# ```

# %%
ranges = {
    "DAPI":  (  20, 1000),
    "FITC":  (1000, 5000),
    "YFP":   (  20,  200),
    "TRITC": (1000, 8000),
}

channel = "DAPI" # to be turned into a loop 
labels = io.imread(f"data/segmented/FOV-1_{channel}.tiff")
intensities = io.imread(f"data/clean/FOV-1_{channel}.tiff")

idx = []
bbox_z = []
bbox_r = []
bbox_c = []
for prop in measure.regionprops(
    label_image=labels,
    intensity_image=intensities
):
    if prop.area < ranges[channel][0] or prop.area > ranges[channel][1]:
        continue
    
    img = prop.image_intensity
    minimum = img[img>0].min()
    img[img>0] = img[img>0] - minimum
    img = img / img.sum()

    remeasure = measure.regionprops(
        label_image=prop.image,
        intensity_image=img
    )
    assert len(remeasure)==1, "Regionprops should return exactly one region"
    remeasure = remeasure[0]
    centroids = [int(np.round(c)) for c in remeasure.centroid_weighted]
    # pad the image to make sure the PSF is centered
    # wait, should the PSF be centered at centroid or maximum? 
    io.imsave(
        f"data/psf/{filepath.stem}_psf-{prop.label}.tiff",
        util.img_as_float32(img)
    )


# %%
