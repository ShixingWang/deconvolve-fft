# %% 
import numpy as np
from skimage import io,util,morphology

image = io.imread("data/clean/FOV-1_DAPI.tiff")

seed = np.copy(image)
seed[1:-1,1:-1,1:-1] = image.min()
mask = image

reconstructed = morphology.reconstruction(seed,mask,method='dilation')

peaks = image - reconstructed