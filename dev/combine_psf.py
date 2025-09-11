# PREVIOUS: ./locate_psf.py
# %%
import deconvolve_fft
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,measure

# %%
def unify_psfs(fov,channel):
    half_window = {
        "DAPI":  10,
        "FITC":  32,
        "YFP":   10,
        "TRITC": 32,
    }

    clean = io.imread(f"data/dev/clean/FOV-{fov}_{channel}.tiff")
    pins = np.loadtxt(f"data/dev/pins/FOV-{fov}_{channel}.txt",dtype=int)
    z_total = clean.shape[0]

    indices = []
    centered_z0 = []
    centered_z1 = []
    centered_tot_z = []
    normalizeds = {}
    for p,(r_max,c_max) in enumerate(pins):
        half_win = half_window[channel]
        if r_max < half_win or r_max > clean.shape[1]-half_win or c_max < half_win or c_max > clean.shape[2]-half_win:
            print(f"Ignored PSF too close to edge: FOV-{fov}_{channel}-{p} at ({r_max},{c_max})")
            continue

        z_max = np.argmax(clean[:,r_max,c_max])

        maximum = clean[z_max,r_max,c_max]

        normalized = np.zeros((z_total, 2*half_win, 2*half_win),dtype=float)
        normalized[:] = clean[:, (r_max-half_win):(r_max+half_win), (c_max-half_win):(c_max+half_win)]
        normalized = normalized / maximum
        normalizeds[p] = normalized

        [ [pad_center_z0,pad_center_z1],_,_,] = deconvolve_fft.calculate_pad4centroid([clean.shape[0],0,0], [z_max,0,0])
        centered_z0.append(pad_center_z0)
        centered_z1.append(pad_center_z1)

        centered_tot_z.append(pad_center_z0 + z_total + pad_center_z1)
        indices.append(p)

    bbox_data = pd.DataFrame({
        "label": indices,
        "centered_z0": centered_z0,
        "centered_z1": centered_z1,
        "centered_tot_z": centered_tot_z,
    })
    max_z = bbox_data['centered_tot_z'].max()
    bbox_data.set_index("label",inplace=True)

    # padding the PSFs to the same size
    psfs = {}
    for idx in indices:
        # pad all psfs to have the same size
        dims_z = bbox_data.loc[idx,'centered_tot_z']
        [ [pad_size_z0,pad_size_z1],_,_ ] = deconvolve_fft.calculate_pad2align([[dims_z,0,0],[max_z,0,0]])[0]

        psf = np.pad(
            normalizeds[idx],
            (
                (bbox_data.loc[idx,"centered_z0"]+pad_size_z0, bbox_data.loc[idx,"centered_z1"]+pad_size_z1),
                (0,0),
                (0,0),
            )
        )
        psf[psf < 0] = 0
        psfs[idx] = psf/psf.max()
    return psfs

# %% 
for channel in (
    # "DAPI",
    "FITC","TRITC",
    # "YFP",
):
    for v in (1,2):
        psfs = unify_psfs(v,channel)
        psfs_keys = list(psfs.keys())
        psfs_array = np.empty( (len(psfs_keys), *psfs[psfs_keys[0]].shape), dtype=float)
        for i,idx in enumerate(psfs_keys):
            psfs_array[i] = psfs[idx]
            io.imsave(
                f"data/dev/psf_individual/psf_FOV-{v}_{channel}-{idx}.tiff",
                util.img_as_float32(psfs[idx])
            )
        psf_median = np.median(psfs_array,axis=0)
        io.imsave(
            f"data/psf/psf-median_FOV-{v}_{channel}.tiff",
            util.img_as_float32(psf_median)
        )

# %%
