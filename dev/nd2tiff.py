import nd2
from pathlib import Path
from skimage import io,util

if not Path("data/dev/tiff").is_dir():
    Path.mkdir("data/dev/tiff")

for path in Path("data/dev/raw/2025-05-13_microspheresOnPetriDish").glob("*.nd2"):
    raw = nd2.imread(str(path))
    print(raw.shape, path.stem)
    io.imsave(
        f"data/dev/tiff/{path.stem}.tiff",
        util.img_as_uint(raw)
    )

# NEXT: ./clean_background.py
