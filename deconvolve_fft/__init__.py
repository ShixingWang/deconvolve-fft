import numpy as np
from skimage import io,util
from scipy import fft

def calculate_pad4centroid(shape,centroid):
    shape_original = np.array(shape)
    assert len(shape_original)==len(centroid), "Centroid has Different Dimensions from Image."

    shape_required = np.array(tuple( int(np.round(2*c)) for c in centroid ))
    pad_lower  = shape_original - shape_required
    pad_lower[pad_lower<0] = 0
    pad_higher = shape_required - shape_original
    pad_higher[pad_higher<0] = 0
    pad_window = [ [pl,ph] for pl,ph in zip(pad_lower,pad_higher) ]

    return pad_window

def center1image(image,centroid):
    pad_window = calculate_pad4centroid(image.shape,centroid)
    return np.pad(image, pad_width=pad_window)


def calculate_pad2align(shapes):
    for shape in shapes[1:]:
        assert len(shape)==len(shapes[0]), "Some Image has Different Dimensions."
    shape_target = np.max(np.array(shapes),axis=0)

    pads = []
    for shape in shapes:
        pad = []
        for shape_i,shapeTi in zip(shape,shape_target):
            pad_0 = (shapeTi - shape_i)//2
            pad_1 = (shapeTi - shape_i) - pad_0
            pad.append([pad_0,pad_1])
        pads.append(pad)
    return pads

def align_images(images):
    """Hope you do not need this function ever LOL."""
    pads = calculate_pad2align([img.shape for img in images])

    aligned_images = []
    for img,pad in zip(images,pads):
        aligned_img = np.pad(img, pad_width=pad)
        aligned_images.append(aligned_img)
    return aligned_images


def _deconvolve(image,psf,epsilon=0.):
    # epsilon ↑: less deconvolved, more similar to original images
    # epsilon ↓: more deconvolved, could give empty images
    # epsilon = 1E-5 is good enough for FITC and TRITC.
    # epsilon = 1E-2 is pretty good for DAPI and YFP.

    fft_img = fft.rfftn(image)
    fft_psf = fft.rfftn(psf)

    fft_obj = fft_img * np.conjugate(fft_psf)/(np.abs(fft_psf)**2+epsilon)
    return fft.ifftshift(fft.irfftn(fft_obj))

def deconvolve(image,psf,epsilon=0.):
    pad1,pad2 = calculate_pad2align([image.shape, psf.shape])
    image = np.pad(image, pad_width=pad1)
    psf   = np.pad(psf,   pad_width=pad2)
    deconvolved = _deconvolve(image,psf,epsilon)
    deconvolved = deconvolved[tuple([slice(p[0],s-p[1]) for p,s in zip(pad1,image.shape)])]
    return deconvolved
