# Find PSFs by Cleaning Background, Segment Each Bead, Crop Around Each Bead, and Average

Order of scripts:
1. `clean_background.py`
2. `find_beads_threshold.py`
3. `deconvolve_beads.py`

The rest of the scripts are side ideas that are not used eventually.

This pipeline gets archived because 
- the segmentation of the beads after cleaning the background looks not as good as the pre-captured single-z image of the beads.
- cleaning of the background introduces some black compensation around bright beads that should not be there.