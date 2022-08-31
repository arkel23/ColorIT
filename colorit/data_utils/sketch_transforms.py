import cv2
import numpy as np
from PIL import Image
from PIL.ImageEnhance import Color, Contrast
from skimage.filters import gaussian


def desaturate(img, percent):
    converter = Color(img)
    img = converter.enhance(percent)
    return img


def decontrast(img, percent):
    converter = Contrast(img)
    img = converter.enhance(percent)
    return img


def xdog(img, sigma=0.8, k=1.6, gamma=0.98, eps=-0.1, phi=200, thresh=False):
    '''
    https://github.com/aaroswings/XDoG-Python/blob/main/XDoG.py
    sigma=0.8, k=1.6, gamma=0.98, eps=-0.1, phi=200

    https://github.com/heitorrapela/xdog/blob/master/main.py
    sigma=0.5, k=1.6, gamma=1, eps=1, phi=1
    these values do not work and lead to all black results (designed for uint8)

    https://subscription.packtpub.com/book/data/9781789537147/1/ch01lvl1sec06/creating-pencil-sketches-from-images
    sigma=0.5, k=200, gamma=1, eps=0.01, phi=10
    these values do get edges but does not look like a sketch or manga
    '''
    img = np.array(img.convert('RGB'))

    g_filtered_1 = gaussian(img, sigma, channel_axis=2)
    g_filtered_2 = gaussian(img, sigma * k, channel_axis=2)

    z = g_filtered_1 - gamma * g_filtered_2

    z[z < eps] = 1.

    mask = z >= eps
    z[mask] = 1. + np.tanh(phi * z[mask])

    if thresh:
        mean = z.mean()
        z[z < mean] = 0.
        z[z >= mean] = 1.

    z = cv2.normalize(src=z, dst=None, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(z.astype('uint8'), 'RGB')


def xdog_serial(img, sigma=0.5, k=4.5, gamma=19, eps=0.01, phi=10**9):
    '''
    https://github.com/SerialLain3170/Colorization/blob/c920440413429af588e0b6bd6799640d1feda68e/nohint_pix2pix/xdog.py
    sigma_range=[0.3, 0.4, 0.5], k_sigma=4.5, p=19, eps=0.01, phi=10**9,
    sigma_large = sigma * k_sigma
    p is similar to gamma but also multiplies by first gaussian
    '''
    img = np.array(img.convert('RGB'))

    g_filtered_1 = gaussian(img, sigma, channel_axis=2)
    g_filtered_2 = gaussian(img, sigma * k, channel_axis=2)

    z = (1+gamma) * g_filtered_1 - gamma * g_filtered_2

    si = np.multiply(img, z)

    edges = np.zeros(si.shape)
    si_bright = si >= eps
    si_dark = si < eps
    edges[si_bright] = 1.0
    edges[si_dark] = 1.0 + np.tanh(phi * (si[si_dark] - eps))

    edges = cv2.normalize(src=edges, dst=None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(edges.astype('uint8'), 'RGB')
