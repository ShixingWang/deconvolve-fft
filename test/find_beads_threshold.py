# %% [markdown]
# The previous 2 scripts tried to first find the local maxima and then its neighboring PSFs.
# This script tries to segment the PSFs first, and find the centroids of them.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# %%  This script helps find the size ranges of PSFs in each channel. 
# ```python
filepath = Path("data/segmented/FOV-2_FITC.tiff")
mask = io.imread(str(filepath))
intensities = io.imread(f"data/clean/{filepath.stem}.tiff")
label_image = measure.label(mask)
props_table = pd.DataFrame(measure.regionprops_table(
    label_image=label_image,
    intensity_image=intensities,
    properties=('label','area','centroid_weighted','bbox')
))
selected_table = props_table[
    props_table["area"].gt(20)
  & props_table["area"].lt(5000)
]

# ```

# %%
def extract_psf(fov,channel):
    # internal parameters
    ranges = {
        "DAPI":  (  20, 1000),
        "FITC":  (1000, 5000),
        "YFP":   (  20,  200),
        "TRITC": (1000, 8000),
    }
    # load inputs
    binary = io.imread(f"data/segmented/FOV-{fov}_{channel}.tiff")
    labels = measure.label(binary)
    intensities = io.imread(f"data/clean/FOV-{fov}_{channel}.tiff")

    # calculate the amount to pad to center the PSFs
    indices = []
    centered_z0 = []
    centered_z1 = []
    centered_r0 = []
    centered_r1 = []
    centered_c0 = []
    centered_c1 = []
    centered_tot_z = []
    centered_tot_r = []
    centered_tot_c = []
    cropped = {}
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
        cropped[prop.label] = img

        remeasure = measure.regionprops(
            label_image=util.img_as_ubyte(prop.image),
            intensity_image=img
        )
        assert len(remeasure)==1, "Regionprops should return exactly one region"
        remeasure = remeasure[0]
        required = tuple(int(np.round(2 * c)) for c in remeasure.centroid_weighted)

        pad_center_left  = np.array(img.shape) - np.array(required)
        pad_center_left[pad_center_left<0] = 0
        pad_center_z0,pad_center_r0,pad_center_c0 = pad_center_left
        pad_center_right = np.array(required) - np.array(img.shape)
        pad_center_right[pad_center_right<0] = 0
        pad_center_z1,pad_center_r1,pad_center_c1 = pad_center_right
        centered_z0.append(pad_center_z0)
        centered_z1.append(pad_center_z1)
        centered_r0.append(pad_center_r0)
        centered_r1.append(pad_center_r1)
        centered_c0.append(pad_center_c0)
        centered_c1.append(pad_center_c1)

        bbox_z,bbox_r,bbox_c = img.shape
        centered_tot_z.append(pad_center_z0 + bbox_z + pad_center_z1)
        centered_tot_r.append(pad_center_r0 + bbox_r + pad_center_r1)
        centered_tot_c.append(pad_center_c0 + bbox_c + pad_center_c1)
        indices.append(prop.label)
        # centereds[prop.label] = centered

    bbox_data = pd.DataFrame({
        "label": indices,
        "centered_z0": centered_z0,
        "centered_z1": centered_z1,
        "centered_r0": centered_r0,
        "centered_r1": centered_r1,
        "centered_c0": centered_c0,
        "centered_c1": centered_c1,
        "centered_tot_z": centered_tot_z,
        "centered_tot_r": centered_tot_r,
        "centered_tot_c": centered_tot_c,
    })
    max_z = bbox_data['centered_tot_z'].max()
    max_r = bbox_data['centered_tot_r'].max()
    max_c = bbox_data['centered_tot_c'].max()
    bbox_data.set_index("label",inplace=True)

    # padding the PSFs to the same size
    psfs = {}
    for idx in indices:
        # pad all psfs to have the same size
        dims_z,dims_r,dims_c = bbox_data.loc[
                                            idx,
                                            ['centered_tot_z','centered_tot_r','centered_tot_c']
                                        ]
        pad_size_z0 = (max_z - dims_z)//2
        pad_size_z1 = (max_z - dims_z) - pad_size_z0
        pad_size_r0 = (max_r - dims_r)//2
        pad_size_r1 = (max_r - dims_r) - pad_size_r0
        pad_size_c0 = (max_c - dims_c)//2
        pad_size_c1 = (max_c - dims_c) - pad_size_c0

        psfs[idx] = np.pad(
            cropped[idx],
            (
                (bbox_data.loc[idx,"centered_z0"]+pad_size_z0, bbox_data.loc[idx,"centered_z1"]+pad_size_z1),
                (bbox_data.loc[idx,"centered_r0"]+pad_size_r0, bbox_data.loc[idx,"centered_r1"]+pad_size_r1),
                (bbox_data.loc[idx,"centered_c0"]+pad_size_c0, bbox_data.loc[idx,"centered_c1"]+pad_size_c1),
            )
        )
    return psfs

# %% 
for channel in ("DAPI","FITC","YFP","TRITC"):
    for v in (1,2):
        psfs = extract_psf(v,channel)

        count_psf = 0
        psf_average = np.zeros_like(psfs[list(psfs.keys())[0]])
        for idx in psfs.keys():
            io.imsave(
                f"data/psf/psf_FOV-{v}_{channel}_idx-{idx}.tiff",
                util.img_as_float32(psfs[idx])
            )
            psf_average += psfs[idx]
            count_psf += 1
        psf_average = psf_average/count_psf
        io.imsave(
            f"data/psf/psf-average_FOV-{v}_{channel}.tiff",
            util.img_as_float32(psf_average)
        )

# %%
