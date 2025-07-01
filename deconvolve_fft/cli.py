import argparse

def deconv1img():
    parser = argparse.ArgumentParser(description="Deconvolve a single image using Fourier Transform.")
    parser.add_argument("--psf",     type=str,   required=True, help="Path to the Point Spread Function (PSF) image.")
    parser.add_argument("--input",   type=str,   required=True, help="Path to the input image file.")
    parser.add_argument("--output",  type=str,   required=True, help="Path to save the deconvolved image.")
    parser.add_argument("--epsilon", type=float, default=1E-6,  help="Tikhonov regularization parameter to avoid division by zero.")
    args = parser.parse_args()

    print("this is deconv1img.")
    # Here you would call the deconvolution function with the provided arguments
    # For example:
    # deconvolve_fft(args.input_image, args.output_image, args.psf, args.iterations)

def deconv1folder():
    parser = argparse.ArgumentParser(description="Deconvolve all images in a folder using Fourier Transform.")
    parser.add_argument("--psf",     type=str,   required=True, help="Path to the Point Spread Function (PSF) image.")
    parser.add_argument("--input",   type=str,   required=True, help="Path to the input folder containing images, if --glob is not set, all files in it will be iterated over.")
    parser.add_argument("--glob",    type=str,   default="*.tiff",  help="Glob pattern to match the files in the folder to deconvolve. Default is '*.tiff'. ")
    parser.add_argument("--output",  type=str,                      help="Path to the output folder. Input folder will be used if not set.")
    parser.add_argument("--prefix",  type=str,   default="deconv_", help="Name prepended to the output file names. Default is 'deconv_'.")
    parser.add_argument("--epsilon", type=float, default=1E-6,      help="Tikhonov regularization parameter to avoid division by zero.")
    args = parser.parse_args()

    print("this is deconv1folder.")
    # Here you would call the deconvolution function for each image in the folder
    # For example:
    # for image_file in os.listdir(args.input_folder):
    #     deconvolve_fft(os.path.join(args.input_folder, image_file), os.path.join(args.output_folder, image_file), args.psf, args.iterations)