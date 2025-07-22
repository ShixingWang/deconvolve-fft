# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,measure,morphology,filters

# %%
for filepath in Path("data/clean").glob("FOV-*.tiff"):
    img = io.imread(str(filepath))
    binary = np.empty_like(img, dtype=bool)
    for z in range(img.shape[0]):
        threshold = filters.threshold_niblack(img[z], window_size=251, k=-1.5)
        binary[z] = (img[z] > threshold)
    binary = morphology.binary_opening(binary, morphology.ball(1))
    labeled = measure.label(binary)
    io.imsave(
        f"data/labeled/{filepath.stem}.tiff",
        util.img_as_uint(labeled)
    )


# %%
def extract_psf(fov,channel):
    # internal parameters
    exclude = {
        "1-DAPI":  [ 2,  9, 13,],
        "1-FITC":  [26, 34, 39, 40, 56, 59, 60, 92, 122, 132],
        "1-YFP":   [],
        "1-TRITC": [ 6,  9, 10, 23, 27, ],
        "2-DAPI":  [],
        "2-FITC":  [ 3, 11, 13, 14, 23, 75, 101],
        "2-YFP":   [],
        "2-TRITC": [ 3,  6, 21, 89],
    }
    # load inputs
    labels = io.imread(f"data/labeled/FOV-{fov}_{channel}.tiff")
    intensities = io.imread(f"data/clean/FOV-{fov}_{channel}.tiff")

    # TODO: clear border

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
        if prop.area < 20 or prop.label in exclude[channel]:
            continue
        
        img = prop.image_intensity
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
for channel in (
    "DAPI",
    "FITC",
    "YFP",
    "TRITC"
):
    for v in (1,2):
        psfs = extract_psf(v,channel)

        count_psf = 0
        psf_average = np.zeros_like(psfs[list(psfs.keys())[0]])
        for idx in psfs.keys():
            io.imsave(
                f"data/psf_crop/psf_FOV-{v}_{channel}_idx-{idx}.tiff",
                util.img_as_float32(psfs[idx])
            )
            psf_average += psfs[idx]
            count_psf += 1
        psf_average = psf_average/count_psf
        io.imsave(
            f"data/psf_crop/psf-average_FOV-{v}_{channel}.tiff",
            util.img_as_float32(psf_average)
        )

# %%
