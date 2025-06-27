# DECONVOLVE-FFT

Deconvolve the images with known Point Spread Functions (PSFs), using the Fourier transform method with a Tikhnov regularization.

## Installation

Choose one of the following methods:

### pip + virtualenv(wrapper)

```bash
mkvirtualenv -r requirements.txt deconv
workon deconv
pip install -e .
```

### poetry

```bash
poetry install
```

## Usage

### As a library

The following functions will be available after `import deconvolve_fft`

- `center1image(image,centroid)`: Pad an image to place its centroid at the center.
    - `calculate_pad4centroid(shape,centroid)`: Given the shape of the image and its centroid coordinates, calculate the padding widths to place the centroid in the center.
- `align_images(images)`: 
    - `calculate_pad2align(shapes)`: Given a list of the shapes of images, calculate the padding widths to 
- `deconvolve(image,psf,epsilon)`: Deconvolve the image with the PSF, after padding both to the same size.
    - `epsilon`: the Tikhnov regularization parameter. The higher this value is, the less deconvolved the product image will be.

### CLI: individual image

- [] Work in progress.

### CLI: iterate over a folder

- [] Work in progress.
