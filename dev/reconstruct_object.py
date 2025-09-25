# PREVIOUS: ./validate_beads.py
# Given the deconvolved bead images (that are not perfect),
# we want to reconstruct the beads by segmenting the images deconvolved with 
# different epsilon values, and the combine the segmentations to get beads. 
# Afterwards, we can get PSF by devonvolve the image with the bead objects.

import numpy as np
from pathlib import Path
from skimage import io,util

channels = (
            "DAPI",
            "FITC",
            "YFP",
            "TRITC"
            )
FoVs = (1,2)

for c in channels:
    for f in FoVs:
        masks = []
        for path in Path("data/validate/fft").glob(f"FOV-{f}_{c}*.tiff"):
            deconvolved = io.imread(str(path))
            mean = deconvolved.mean()
            std  = deconvolved.std()
            threshold = mean + 3*std
            masks.append( (deconvolved > threshold) )
        # TODO: stack masks and do majority voting