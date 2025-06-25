import numpy as np
from skimage import io,util
from scipy import fft

def center1image(image,centroid):
    shape_original = np.array(image.shape)
    assert len(shape_original)==len(centroid), "Centroid has Different Dimensions from Image."

    shape_required = np.array(tuple( int(np.round(2*c)) for c in centroid ))
    pad_lower  = shape_original - shape_required
    pad_lower[pad_lower<0] = 0
    pad_higher = shape_required - shape_original
    pad_higher[pad_higher<0] = 0
    pad_window = [ [pl,ph] for pl,ph in zip(pad_lower,pad_higher) ]

    return np.pad(image, pad_width=pad_window)

def align2images(img1,img2):
    shape1 = np.array(img1.shape)
    shape2 = np.array(img2.shape)
    assert len(shape1)==len(shape2), "Two Images have Different Dimensions."
    shape_target = np.max(np.vstack([shape1,shape2]),axis=0)

    pad1 = []
    pad2 = []
    for shape1i,shape2i,shapeTi in zip(shape1,shape2,shape_target):
        pad10 = (shapeTi - shape1i)//2
        pad11 = (shapeTi - shape1i) - pad10
        pad1.append([pad10,pad11])

        pad20 = (shapeTi - shape2i)//2
        pad21 = (shapeTi - shape2i) - pad20
        pad2.append([pad20,pad21])

    aligned1 = np.pad(img1, pad_width=pad1)
    aligned2 = np.pad(img2, pad_width=pad2)
    return aligned1,aligned2

def deconvolve(image,psf,epsilon=0.):
    # epsilon ↑: less deconvolved, more similar to original images
    # epsilon ↓: more deconvolved, could give empty images
    # epsilon = 1E-5 is good enough for FITC and TRITC.
    # epsilon = 1E-2 is pretty good for DAPI and YFP.

    fft_img = fft.rfftn(image)
    fft_psf = fft.rfftn(psf)

    fft_obj = fft_img * np.conjugate(fft_psf)/(np.abs(fft_psf)**2+epsilon)
    return fft.ifftshift(fft.irfftn(fft_obj))
