# before this script, I cleaned the large stains in the cleaned images
# %%
import numpy as np
from pathlib import Path
from skimage import io,util,measure

# %%
def construct_beads(intensities,labels):
    beads = np.zeros_like(intensities, dtype=int)
    for prop in measure.regionprops(
            label_image=labels,
        intensity_image=intensities
    ):
        if prop.area < 20:
            continue
        intensity = prop.image_intensity.sum()
        if np.isnan(intensity):
            continue
        centroid = np.array(prop.centroid_weighted)
        if np.any(centroid < 0) or np.any(centroid >= intensities.shape):
            print(f"Warning: centroid {centroid} out of bounds for shape {intensities.shape}")
            continue
        start = centroid.astype(int)
        delta = centroid - start
        dims = len(intensities.shape)
        for i in range(2**dims):
            shift = np.array(list(np.binary_repr(i,width=dims))).astype(int)
            coord = tuple(start + shift)
            beads[coord] = int(intensity * np.prod(
                np.ones(dims) / 2
              - (-1)**(shift+1) / 2
              - (-1)**shift * delta
            ))
    return beads

for path_clean in Path("data/clean").glob("FOV-*.tiff"):
    path_labeled = Path("data/labeled") / path_clean.name

    intensities = io.imread(str(path_clean))
    labels      = io.imread(str(path_labeled))
    # if np.isnan(intensities).any():
    #     print(f"Warning: {path_clean.name} has NaN values, skipping")
    # if np.isnan(labels).any():
    #     print(f"Warning: {path_labeled.name} has NaN values, skipping")
        
    beads = construct_beads(intensities,labels)
    if beads.max() > 65535:
        beads = (65535 * beads/beads.max()).astype(np.uint16)
        print(f"Warning: {path_clean.name} has values > 65535, rescaling to uint16")
    io.imsave(
        f"data/psf_deconv/beads_{path_clean.name}",
        util.img_as_uint(beads)
    )

# %%
from scipy import fft

def deconvolve(image,psf,epsilon=0.):
    # epsilon ↑: less deconvolved, more similar to original images
    # epsilon ↓: more deconvolved, could give empty images
    # epsilon = 1E-5 is good enough for FITC and TRITC.
    # epsilon = 1E-2 is pretty good for DAPI and YFP.

    fft_img = fft.rfftn(image)
    fft_psf = fft.rfftn(psf)

    fft_obj = fft_img * np.conjugate(fft_psf)/(np.abs(fft_psf)**2+epsilon)
    return fft.ifftshift(fft.irfftn(fft_obj))

for path_clean in Path("data/clean").glob("FOV-*.tiff"):
    path_beads = Path("data/psf_deconv") / f"beads_{path_clean.name}"

    cleaned = io.imread(str(path_clean))
    beads   = io.imread(str(path_beads))
    psf = deconvolve(cleaned, beads, epsilon=1E-10)
    psf = psf[:,]
    io.imsave(
        f"data/psf_deconv/psf-deconv_{path_clean.name}",
        util.img_as_float32(psf)
    )

# %%
