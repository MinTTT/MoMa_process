
"""
rotation.py contains a find_rotation orientation detection function, a rotate_image function which will use the
fastest available library call, and various helper functions to calculate crop zones of rotated images.
"""

# from __future__ import division, unicode_literals, print_function


from .signal import find_phase, vertical_mean, remove_outliers, each_image_slice, hamming_smooth
# from ..test import test_image
import math
import numpy as np
from joblib import Parallel, delayed

def find_rotation(image, steps=10, smoothing_signal_length=15):
    """
    Tries to detect the rotation by pairwise cross-correlation of vertical mean profiles of the image.
    The image is split into ``steps`` slices. Smoothing of intermediate signals is performed with a
    `smoothing_signal_length`-wide Hamming window.

    :param image: input image
    :param steps: step count
    :param smoothing_signal_length: int
    :type image: numpy.ndarray
    :type steps: int
    :type smoothing_signal_length: int
    :return: angle: float
    :rtype: float

    >>> find_rotation(test_image())
    -1.5074357587749678
    """

    shifts = np.zeros(steps)

    last_signal = None
    last_fft = None

    step = 0

    all_profiles = []

    for n, the_step, image_slice in each_image_slice(image, steps, direction='vertical'):
        step = the_step
        profile = vertical_mean(image_slice)

        profile = hamming_smooth(profile, smoothing_signal_length)
        profile = np.diff(profile)

        all_profiles.append(profile)

        if n == 0:
            last_signal = profile
            last_fft = None
            continue

        shift, current_fft = find_phase(last_signal, profile, last_fft, return_2=True)
        last_signal, last_fft = profile, current_fft

        shifts[n] = shift

    shifts = remove_outliers(shifts)

    return math.atan(np.mean(shifts) / step) * 180.0 / math.pi


try:
    # noinspection PyUnresolvedReferences
    import cv2

    def rotate_image(image, angle):
        """
        Rotates image for angle degrees. Shape remains the same.

        :param image: input image
        :param angle: angle to rotate
        :type image: numpy.ndarray
        :type angle: float
        :rtype: numpy.ndarray
        :return: rotated image

        >>> rotate_image(np.array([[1, 0, 0, 0],
        ...                        [0, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [0, 0, 0, 1]], dtype=np.uint8), 45.0)
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [0, 0, 0, 0]], dtype=uint8)
        """

        return cv2.warpAffine(image,
                              cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), angle, 1.0),
                              (image.shape[1], image.shape[0]))

except ImportError:
    # DO NOT USE from scipy.misc import imrotate
    from scipy.ndimage.interpolation import rotate

    def rotate_image(image, angle):
        """

        :param image:
        :param angle:
        :return:
        """
        return rotate(image, angle=angle, reshape=False)


def calculate_crop_for_angle(image, angle):
    """
    Calculates the pixels in vertical and horizontal direction which become invalid, i.e.,
    not completely filled with image.

    :param image: input image
    :param angle: rotation angle
    :type image: numpy.ndarray
    :type angle: float
    :return: (vertical_crop, horizontal_crop)
    :rtype: tuple(int, int)

    >>> calculate_crop_for_angle(np.zeros((32, 32,)), 45.0)
    (15, 15)
    """
    wd = (image.shape[0] * 0.5) * math.tan(angle / (180.0 / math.pi))
    hd = (image.shape[1] * 0.5) * math.tan(angle / (180.0 / math.pi))
    hd, wd = int(abs(hd)), int(abs(wd))
    return hd, wd


def apply_rotate_and_cleanup(image, angle):
    """
    Rotates image for angle degrees, and crops the result to only return defined contents.

    :param image: input image
    :param angle: angle to rotate
    :type image: numpy.ndarray
    :type angle: float
    :return: the rotated and cropped image, the angle, the horizontal crop, the vertical crop
    :rtype: tuple(numpy.ndarray, float, int, int)

    >>> apply_rotate_and_cleanup(np.zeros((32, 32,)), 45.0)
    (array([[ 0.,  0.],
           [ 0.,  0.]]), 45.0, 15, 15)
    """
    new_image = rotate_image(image, angle)
    h, w = calculate_crop_for_angle(image, angle)
    lh, rh = (h, -h) if h else (None, None)
    lw, rw = (w, -w) if w else (None, None)
    new_image = new_image[lh:rh, lw:rw]
    return new_image, angle, h, w

def crop_images(im, ang):
    im = apply_rotate_and_cleanup(im, ang)[0]
    return im

def rotate_fov(ims, crop=True):
    '''
    This function used parallel style to rotate images, made side channels vertical.
    :param ims: images array, 3D (fov, y, x)
    :param crop: bool, default True.
        if True, images will crop the invalid border, images size will change.
    :return:
    '''
    num_fov = len(ims)
    rot_angles = Parallel(n_jobs=30)(delayed(find_rotation)(ims[idx_fov, ...]) for idx_fov in range(num_fov))
    if crop:
        rot_ims = Parallel(n_jobs=30)(delayed(crop_images)(ims[idx_fov, ...], rot_angles[idx_fov])
                                      for idx_fov in range(num_fov))
    else:
        rot_ims = Parallel(n_jobs=30)(delayed(rotate_image)(ims[idx_fov, ...], rot_angles[idx_fov])
                                      for idx_fov in range(num_fov))
    return np.array(rot_ims), np.array(rot_angles)
