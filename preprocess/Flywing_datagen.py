#!/usr/bin/env python
# coding: utf-8

# <hr style="height:2px;">
# 
# # Demo: Training data generation for joint denoising and surface projection of *Drosophila melanogaster* wing
# 
# This notebook demonstrates training data generation for a 3D → 2D denoising+projection task, where corresponding pairs of low and high quality 3D stacks can be acquired. The surface of interest is then extracted from the high quality stacks with a conventional projection method, such as [PreMosa](https://doi.org/10.1093/bioinformatics/btx195).
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

# In[ ]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches_reduced_target


# <hr style="height:2px;">
# 
# # Download example data
# 
# First we download some example data, consisting of low-SNR 3D stacks with corresponding 2D surface images extracted from the high-SNR stacks.  
# Note that `GT` stands for [ground truth](https://en.wikipedia.org/wiki/Ground_truth) and represents high signal-to-noise ratio (SNR) images.

# In[ ]:


download_and_extract_zip_file (
    url       = 'http://csbdeep.bioimagecomputing.com/example_data/flywing.zip',
    targetdir = 'data',
)


# We can plot one of the training pairs:

# In[ ]:


y = imread('data/flywing/GT/session_4_P08.tif')
x = imread('data/flywing/low_C0/session_4_P08.tif')
print('input  image size =', x.shape)
print('output image size =', y.shape)

plt.figure(figsize=(16,10))
plot_some(np.stack([x,np.broadcast_to(y,x.shape)]),
          title_list=[['low-SNR stack (maximum projection)','high-SNR (surface extracted with PreMosa)']], 
          pmin=2,pmax=99.8)


# <hr style="height:2px;">
# 
# # Generate training data for denoising + projection CARE
# 
# We first need to create a `RawData` object, which defines how to get the pairs of low/high SNR images and the semantics of each axis (e.g. which one is considered a color channel, etc.).
# 
# Here we have several folders with low-SNR images and one folder "GT" with the high-SNR extracted surface images. Note that corresponding images are TIFF files with identical names.  
# For this case, we use `RawData.from_folder` and set `axes = 'ZYX'` to indicate the semantic axes of the low-SNR input stacks.

# In[ ]:


raw_data = RawData.from_folder (
    basepath    = 'data/flywing',
    source_dirs = ['low_C0','low_C2','low_C3'],
    target_dir  = 'GT',
    axes        = 'ZYX',
)


# From corresponding images, we now generate some 3D/2D patches. As a general rule, use a `patch_size` that is a power of two along all non-channel axes, here at least divisible by 16. You can use `None` along the projection axis (typically `Z`, i.e. use `reduction_axes = 'Z'`) to indicate that each patch should contain the entire image along this axis.
# Furthermore, set `target_axes` appropriately if the target images are missing the projection axis.
# 
# Note that returned values `(X, Y, XY_axes)` by `create_patches_reduced_target` are not to be confused with the image axes X and Y.  
# By convention, the variable name `X` (or `x`) refers to an input variable for a machine learning model, whereas `Y` (or `y`) indicates an output variable.

# In[ ]:


X, Y, XY_axes = create_patches_reduced_target (
    raw_data            = raw_data,
    patch_size          = (None,128,128),
    n_patches_per_image = 16,
    target_axes         = 'YX',
    reduction_axes      = 'Z',
    save_file           = 'data/my_training_data.npz',
)


# In[ ]:


print("shape of X   =", X.shape)
print("shape of Y   =", Y.shape)
print("axes  of X,Y =", XY_axes)


# ## Show
# 
# This shows some of the generated patch pairs (odd rows: maximum projection of *source*, even rows: *target*)

# In[ ]:


for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
None

