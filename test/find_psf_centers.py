# %% [markdown]
# What we have:
# - `data/2022-08-31_Beads`: 37*2048*2044
#     + dilution-0:   no single pixel dots
#     + dilution-20:  contains signals, but backgournd not uniform
#     + dilution-100_field-2: looks good enough, but only this one
# - `data/2022-11-08_beads_equal_pixel_size`: could not see in the blue channels
# - `data/2025-05-13_microspheresOnPetriDish`: dots are there, but background is a problem


# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import io,feature

# %% before 
