

import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib import cm
from matplotlib.pyplot import bar
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import fftpack
# Convolution:
from scipy.signal import convolve2d
from skimage.color import rgb2gray, rgb2hsv
from skimage.exposure import histogram
from skimage.feature import canny
# Edges
from skimage.filters import (median, prewitt, roberts, sobel, sobel_h, sobel_v,
                             threshold_mean)
# from skimage.util import random_noise

from skimage import util


# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

