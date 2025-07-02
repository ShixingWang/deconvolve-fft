import nd2
import argparse
import numpy as np
from pathlib import Path
from skimage import io,util
from deconvolve_fft import deconvolve  # Assuming you have a deconvolve_fft function defined in this module

def deconv1img():
    parser = argparse.ArgumentParser(description="Deconvolve using Fourier Transform a single image.")
    parser.add_argument("--psf",     type=str,   required=True, help="Path to the Point Spread Function (PSF) image.")
    parser.add_argument("--input",   type=str,   required=True, help="Path to the input image file.")
    parser.add_argument("--output",  type=str,   required=True, help="Path to save the deconvolved image.")
    parser.add_argument("--epsilon", type=float, default=1E-6,  help="Tikhonov regularization parameter to avoid division by zero.")
    args = parser.parse_args()

    io_psf = nd2 if args.psf.endswith('.nd2') else io
    img_psf = io_psf.imread(args.psf)

    io_input = nd2 if args.input.endswith('.nd2') else io
    img_input = io_input.imread(args.input)

    deconvolved = deconvolve(img_input, img_psf, epsilon=args.epsilon)
    format_function = util.img_as_uint if np.issubdtype(img_input.dtype, np.integer) else util.img_as_float32
    io.imsave(
        args.output, 
        format_function(deconvolved)
    )
    return None

def deconv1folder():
    parser = argparse.ArgumentParser(description="Deconvolve using Fourier Transform all images matching a glob pattern in a folder.")
    parser.add_argument("--psf",     type=str,   required=True, help="Path to the Point Spread Function (PSF) image.")
    parser.add_argument("--input",   type=str,   required=True, help="Path to the input folder containing images, if --glob is not set, all files in it will be iterated over.")
    parser.add_argument("--glob",    type=str,   default="*.tiff",  help="Glob pattern to match the files in the folder to deconvolve. Default is '*.tiff'. ")
    parser.add_argument("--output",  type=str,                      help="Path to the output folder. Input folder will be used if not set.")
    parser.add_argument("--prefix",  type=str,   default="deconv_", help="Name prepended to the output file names. Default is 'deconv_'.")
    parser.add_argument("--epsilon", type=float, default=1E-6,      help="Tikhonov regularization parameter to avoid division by zero.")
    args = parser.parse_args()

    io_psf = nd2 if args.psf.endswith('.nd2') else io
    img_psf = io_psf.imread(args.psf)

    for path_image in Path(args.input).glob(args.glob):
        io_img = nd2 if path_image.suffix == '.nd2' else io
        image = io_img.imread(args.input)
        
        deconvolved = deconvolve(image, img_psf, epsilon=args.epsilon)
        format_function = util.img_as_uint if np.issubdtype(image.dtype, np.integer) else util.img_as_float32
        path_output = f"{args.output if args.output else args.input}/{args.prefix}{path_image.stem}.tiff"
        io.imsave(path_output, format_function(deconvolved))
    return None