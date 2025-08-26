# %%
import deconvolve_fft
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,measure

# %%
def get_center(image, mode):
    if   mode == "max":
        center = np.unravel_index(np.argmax(image), image.shape)
    elif mode == "centroid":
        mask = np.ones_like(image, dtype=int)
        measured = measure.regionprops(
            label_image=mask,
            intensity_image=image
        )
        assert len(measured) == 1, "There should be only one region in the image"
        center = measured[0].centroid_weighted
    else: 
        raise ValueError("Mode must be 'max' or 'centroid'")
    return center

def unify_psfs(fov,channel,mode):
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
    normalizeds = {}

    for path in Path("data/dev/psf_crop").glob(f"FOV-{fov}_{channel}*.tif"):
        p = int(path.stem.rpartition("-")[2])
        image = io.imread(str(path))
        
        normalized = np.empty_like(image,dtype=float)
        for z in range(image.shape[0]):
            threshold = np.percentile(image[z],0.1)
            normalized[z] = image[z] - threshold
            # image[image<0] = 0
            # image = image / np.max(image)
        normalized[normalized < 0] = 0
        normalized = normalized / np.max(normalized)
        normalizeds[p] = normalized

        centroid = get_center(normalized, mode)
        
        [
            [pad_center_z0,pad_center_z1],
            [pad_center_r0,pad_center_r1],
            [pad_center_c0,pad_center_c1],
        ] = deconvolve_fft.calculate_pad4centroid(image.shape, centroid)
        centered_z0.append(pad_center_z0)
        centered_z1.append(pad_center_z1)
        centered_r0.append(pad_center_r0)
        centered_r1.append(pad_center_r1)
        centered_c0.append(pad_center_c0)
        centered_c1.append(pad_center_c1)

        bbox_z,bbox_r,bbox_c = image.shape
        centered_tot_z.append(pad_center_z0 + bbox_z + pad_center_z1)
        centered_tot_r.append(pad_center_r0 + bbox_r + pad_center_r1)
        centered_tot_c.append(pad_center_c0 + bbox_c + pad_center_c1)
        indices.append(p)
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
        [
            [pad_size_z0,pad_size_z1],
            [pad_size_r0,pad_size_r1],
            [pad_size_c0,pad_size_c1],
        ] = deconvolve_fft.calculate_pad2align([[dims_z,dims_r,dims_c],[max_z,max_r,max_c]])[0]

        psf = np.pad(
            normalizeds[idx],
            (
                (bbox_data.loc[idx,"centered_z0"]+pad_size_z0, bbox_data.loc[idx,"centered_z1"]+pad_size_z1),
                (bbox_data.loc[idx,"centered_r0"]+pad_size_r0, bbox_data.loc[idx,"centered_r1"]+pad_size_r1),
                (bbox_data.loc[idx,"centered_c0"]+pad_size_c0, bbox_data.loc[idx,"centered_c1"]+pad_size_c1),
            )
        )
        for z in range(psf.shape[0]):
            if np.any(psf[z]>0):
                samples = []
                samples.append(normalizeds[idx][:, 10,:].flatten())
                samples.append(normalizeds[idx][:,-10,:].flatten())
                samples.append(normalizeds[idx][:,:, 10].flatten())
                samples.append(normalizeds[idx][:,:,-10].flatten())
                sample = np.hstack(samples)
            else:
                if z < psf.shape[0]/2:
                    sample = normalizeds[idx][ 0].flatten()
                else:
                    sample = normalizeds[idx][-1].flatten()                
            mu    = np.mean(sample)
            sigma = np.std(sample,ddof=1)
            psf[z,psf[z]==0] = np.random.normal(mu,sigma,size=np.count_nonzero(psf[z]==0))
        psf[psf < 0] = 0
        psfs[idx] = psf/psf.max()
    return psfs

# %% 
for channel in ("DAPI","FITC","TRITC","YFP"):
    for v in (1,2):
        for mode in ["max"]:
            psfs = unify_psfs(v,channel,mode)

            count_psf = 0
            psf_average = np.zeros_like(psfs[list(psfs.keys())[0]])
            for idx in psfs.keys():
                psf_average += psfs[idx]
                count_psf += 1
            psf_average = psf_average/count_psf
            io.imsave(
                f"data/psf/psf-{mode}_FOV-{v}_{channel}.tiff",
                util.img_as_float32(psf_average)
            )

# %%
