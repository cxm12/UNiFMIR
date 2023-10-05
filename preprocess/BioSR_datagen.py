#!/usr/bin/env python
# coding: utf-8

# <hr style="height:2px;">
#
# # Demo: Denoising of 2D cell images
#
# This notebook demonstrates training data generation for a 2D denoising task, where corresponding pairs of low and high quality images can be acquired.
#
# The high SNR images are acquistions of Human U2OS cells taken from the [Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC006/) and the low SNR images were created by synthetically adding *strong read-out and shot-noise* (and additionally applying *pixel binning* of 2x2) thus mimicking acquisitions at a very low light level.
#
# ![](imgs/denoising_binning_overview.png)
#
# Each image pair should be registered, which in a real application setting is best achieved by acquiring both images _interleaved_, i.e. as different channels that correspond to the different exposure/laser settings.
# Since the image pairs were synthetically created in this example, they are already perfectly aligned.
#
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from csbdeep.utils import plot_some
from csbdeep.data import RawData, create_patchesSR
from csbdeep.io import save_training_data, load_training_data
from tifffile import imsave
import os
from PIL import Image
from csbdeep.func_mcx import savecolorim
from csbdeep.utils import normalize


def saveTestSet():
    namelst = ['ER']  # ['F-actin', 'CCPs']  # 'Microtubules'  #
    for name in namelst:
        traindatapath = '../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_training_data.npz' % name
        savepath = '../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/GT/' % name
        savepath1 = '../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/LR/' % name
        os.makedirs(savepath, exist_ok=True)
        os.makedirs(savepath1, exist_ok=True)
        axes = 'SCYX'  #
    
        (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=0.05, axes=axes, verbose=True)
        # (18468, 128, 128, 1) (18468, 256, 256, 1)

        for i in range(100):
            print(i)
            imsave(savepath1 + 'im%d_LR.tif' % (i + 1), X_val[i])
            xn = np.uint8(normalize(np.squeeze(X_val[i]), 0, 100, clip=True) * 255)  # 对输出做normalization
            savecolorim(savepath1 + 'im%d_LR.png' % (i + 1), xn)
            # Image.fromarray(np.uint8(np.squeeze(X_val[i]))).save(savepath1 + 'im%d_LR.png' % (i + 1))
            imsave(savepath + 'im%d_GT.tif' % (i + 1), Y_val[i])
            yn = np.uint8(normalize(np.squeeze(Y_val[i]), 0, 100, clip=True) * 255)  # 对输出做normalization
            savecolorim(savepath + 'im%d_GT.png' % (i + 1), yn)
            # Image.fromarray(np.uint8(np.squeeze(Y_val[i]))).save(savepath + 'im%d_GT.png' % (i + 1))

        # save_training_data('../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_test_data.npz' % name,
        #                    X_val, Y_val, axes)
        # print('save to : ../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_test_data.npz' % name)
    exit()


def saveTrainSet():
    # # Generate training data for CARE
    #
    # We first need to create a `RawData` object, which defines how to get the pairs of low/high SNR images and the semantics of each axis (e.g. which one is considered a color channel, etc.).
    #
    # Here we have two folders "low" and "GT", where corresponding low and high-SNR TIFF images have identical filenames.
    # For this case, we can simply use `RawData.from_folder` and set `axes = 'YX'` to indicate the semantic order of the image axes (i.e. we have typical 2 dimensional images).
    
    name = 'ER'  # 'CCPs'  # 'F-actin'  # 'Microtubules'  #
    raw_data = RawData.from_folder(
        basepath='../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s' % name,
        source_dirs=['training_wf'],
        target_dir='training_gt',
        axes='YX',
    )
    
    # From corresponding images, the function `create_patches` will now generate lots of paired patches that will be used for training the CARE model later.
    #
    # `create_patches` returns values `(X, Y, XY_axes)`.
    # By convention, the variable name `X` (or `x`) refers to an input variable for a machine learning model, whereas `Y` (or `y`) indicates an output variable.
    #
    # As a general rule, use a *patch size* that is a power of two along all axes, or which is at least divisible by 8. For this example we will use patches of size 128x128.
    #
    # An important aspect is *data normalization*, i.e. the rescaling of corresponding patches to a dynamic range of ~ (0,1). By default, this is automatically provided via percentile normalization, which can be adapted if needed.
    #
    # By default, patches are sampled from *non-background regions* i.e. that are above a relative threshold that can be given in the function below. We will disable this for this dataset as most image regions already contain foreground pixels and thus set the threshold to 0.
    
    from csbdeep.data import no_background_patches
    
    X, Y, XY_axes = create_patchesSR(
        raw_data=raw_data,
        patch_size=(128, 128),
        HRpatch_size=(256, 256),
        patch_filter=no_background_patches(0),
        n_patches_per_image=1,
        save_file='../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_training_data.npz' % name
    )
    
    # assert X.shape == Y.shape
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)
    
    # ## Show some example patches
    # This shows some of the generated patch pairs (even rows: *input*, odd rows: *target*)
    
    for i in range(2):
        plt.figure(figsize=(16, 4))
        sl = slice(8 * i, 8 * (i + 1)), 0
        plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.show()


if __name__ == '__main__':
    saveTrainSet()
    saveTestSet()

    # The data set is already split into a **train** and **test** set, each containing low SNR ("low") and corresponding high SNR ("GT") images.

    # We can plot some training images:

    # y = imread('../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/CCPs/training_gt/00000001.tif')
    # x = imread('../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/CCPs/training_wf/00000001.tif')
    # print('image size =', x.shape)
    #
    # plt.figure(figsize=(13,5))
    # plt.subplot(1,2,1)
    # plt.imshow(x, cmap  ="magma")
    # plt.colorbar()
    # plt.title("low")
    # plt.subplot(1,2,2)
    # plt.imshow(y, cmap  ="magma")
    # plt.colorbar()
    # plt.title("high")
    # plt.show()
