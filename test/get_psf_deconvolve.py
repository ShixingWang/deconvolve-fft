import numpy as np
from pathlib import Path
from skimage import io,util,measure

def construct_beads(intensities,labels):
    beads = np.zeros_like(intensities, dtype=int)
    for prop in measure.regionprops(
            label_image=labels,
        intensity_image=intensities
    ):
        if prop.area < 20:
            continue
        intensity = prop.image_intensity.sum()
        centroid = np.array(prop.centroid_weighted)
        start = centroid.astype(int)
        delta = centroid - start
        for i in range(2**len(intensities.shape)):
            shift = np.array(list(np.binary_repr(i,width=len(intensities.shape))))
            coord = tuple(start + shift)
            beads[coord] = intensity * np.prod()
            continue
    return beads



for path_clean in Path("data/clean").glob("FOV-*.tiff"):
    path_labeled = Path("data/labeled") / path_clean.name

    intensities = io.imread(str(path_clean))
    labels      = io.imread(str(path_labeled))
    beads = construct_beads(intensities,labels)


