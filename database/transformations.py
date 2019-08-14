"""
author: Noorvir Aulakh
date: 27 July 2017

"""

import skimage.transform
import scipy.ndimage
import scipy.misc
import numpy as np
import logging
import warnings
import time

import scipy.misc

# Display logging info
logging.getLogger().setLevel(logging.INFO)


class ImageTransform(object):

    def __init__(self):

        pass

    @staticmethod
    def resample(imgs, output_size, interpolation='bicubic'):
        """ Up-sample image(s) using bi-cubic interpolation"""

        output_shape = [np.shape(imgs)[0], output_size, output_size, 1]
        output_imgs = np.zeros(output_shape[:3])
        imgs = np.reshape(imgs, np.shape(imgs)[:3])

        for _id, img in enumerate(imgs):
            output_imgs[_id][:, :] = scipy.misc.imresize(img, output_shape[1:3], interp=interpolation, mode='L')

        # convert to minimum precision to save space
        output_imgs = output_imgs.astype(np.uint8, copy=False)

        return np.reshape(output_imgs, output_shape)

#
# def upsample_1(scale_factor, input_img):
#     # Pad with 0 values, similar to how Tensorflow does it.
#     # Order=1 is bilinear upsampling
#
#
#     # output =  skimage.transform.rescale(input_img,
#     #                                  scale_factor,
#     #                                  mode='constant',
#     #                                  cval=0,
#     #                                  order=1)
#
#     output = scipy.ndimage.zoom(imgs, (1,2,2), order=3)
#     return output
#
# dir = '/home/noorvir/datasets/gqcnn/dexnet_mini/'
# file = 'depth_ims_tf_00000.npz'
#
# img = np.load(dir + file)['arr_0'][0]
# img = np.reshape(img, [32, 32])
# img2 = np.copy(img)
#
# imgs = np.array([img, img2])
# io.imshow(img, interpolation='none')
#
# io.show()
#
# upsampled_imgs = upsample_1(scale_factor=8, input_img=imgs)
#
# for img in upsampled_imgs:
#
#     io.imshow(img, interpolation='none')
#     io.show()