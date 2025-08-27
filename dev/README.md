# Find PSFs by Cleaning Background, Segment Each Bead, Crop Around Each Bead, and Average

`proof_of_concept.py` goes before everything, and is actually independent from the workflow.

Order of scripts (notice that this is different from v-0.1.0):
1. `locate_psf.py`
2. `clean_background.py`
3. `crop_psf.py`
4. `deconvolve_beads.py`

