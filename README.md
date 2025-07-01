# DECONVOLVE-FFT

Deconvolve diffraction-blurred images with known Point Spread Functions (PSFs), using the Fourier transform method with a Tikhonov regularization.

Here a diffraction-blurred image is modeled as a convolution between the object and the PSF of the imaging system. Given the convolution theorem of the Fourier transformation, we can do the Fast Fourier Transform (FFT) on both the image and the PSF, and then calculate the object by the inverse Fourier transform of the quotient. To avoid divergence of division by zero, a little real number (Tikhonov regularization) parameter is added to the denominator.

## Installation

Choose *one* of the following methods you are familiar with:

### conda

```bash
conda create -n deconv
conda activate deconv
pip install --no-deps -r requirements.txt 
pip install -e .
```

### pip + virtualenv(wrapper)

```bash
mkvirtualenv deconv
workon deconv
pip install --no-deps -r requirements.txt 
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
- `align_images(images)`: Pad a list of images so that they are of the same shape. 
    - `calculate_pad2align(shapes)`: Given a list of the shapes of images, calculate the padding widths for each image to be padded into the same size (maximums on each dimension).
- `deconvolve(image,psf,epsilon)`: Deconvolve the image with the PSF which do not have to be of the same size (will be padded by this function).
    - `_deconvolve(image,psf,epsilon)`: Core calculation. User is responsible to make sure `image` and `psf` are of the same size.
    - `epsilon`: the Tikhnov regularization parameter. The higher this value is, the less deconvolved the product image will be.

### CLI: individual image

- [ ] Work in progress.

### CLI: iterate over a folder

- [ ] Work in progress.

## LICENSE 

[MIT license](LICENSE).
