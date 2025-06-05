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
binary = io.imread(f"data/segmented/FOV-1_{channel}.tiff")
labels = measure.label(binary)
intensities = io.imread(f"data/clean/FOV-1_{channel}.tiff")

indices = []
bboxes_z = []
bboxes_r = []
bboxes_c = []
centereds = {}
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
        label_image=util.img_as_ubyte(prop.image),
        intensity_image=img
    )
    assert len(remeasure)==1, "Regionprops should return exactly one region"
    remeasure = remeasure[0]
    
    required = tuple(int(np.round(2 * c)) for c in remeasure.centroid_weighted)
    pad_center_left  = np.array(img.shape) - np.array(required)
    pad_center_left[pad_center_left<0] = 0
    pad_center_right = np.array(required) - np.array(img.shape)
    pad_center_right[pad_center_right<0] = 0
    centered = np.pad(
        img,
        (
            (pad_center_left[0],pad_center_right[0]),
            (pad_center_left[1],pad_center_right[1]),
            (pad_center_left[2],pad_center_right[2]),
        )
    )
    bbox_z,bbox_r,bbox_c = centered.shape
    bboxes_z.append(bbox_z)
    bboxes_r.append(bbox_r)
    bboxes_c.append(bbox_c)
    indices.append(prop.label)
    centereds[prop.label] = centered
    # io.imsave(
    #     f"data/psf/{filepath.stem}_psf-{prop.label}.tiff",
    #     util.img_as_float32(img)
    # )

# %%
bbox_data = pd.DataFrame({
    "label": indices,
    "bbox_z": bboxes_z,
    "bbox_r": bboxes_r,
    "bbox_c": bboxes_c,
})
max_z = bbox_data['bbox_z'].max()
max_r = bbox_data['bbox_r'].max()
max_c = bbox_data['bbox_c'].max()
psf_average = np.zeros((max_z,max_r,max_c))
count_psf = 0
for idx in indices:
    # pad all psfs to have the same size
    dims = centereds[idx].shape
    pad_size_z0 = (max_z - dims[0])//2
    pad_size_z1 = (max_z - dims[0]) - pad_size_z0
    pad_size_r0 = (max_z - dims[1])//2
    pad_size_r1 = (max_z - dims[1]) - pad_size_r0
    pad_size_c0 = (max_z - dims[2])//2
    pad_size_c1 = (max_z - dims[2]) - pad_size_c0
    
    psf = np.pad(
        centereds[idx],
        (
            (pad_size_z0,pad_size_z1),
            (pad_size_r0,pad_size_r1),
            (pad_size_c0,pad_size_c1),
        )
    )
    io.imsave(
        f"data/psf/{filepath.stem}_psf-{idx}.tiff",
        util.img_as_float32(psf)
    )

    psf_average += psf
    count_psf += 1
psf_average = psf_average/count_psf
io.imsave(
    f"data/psf/average_{filepath.stem}.tiff",
    util.img_as_float32(psf_average)
)

# %%
