import numpy as np
import matplotlib.pyplot as plt
from .utils.tf import K, IS_TF_1, tf
if IS_TF_1:
    from skimage.measure import compare_psnr, compare_ssim
else:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
    
    return flops.total_float_ops  # Prints the "flops" of the model.


def compute_psnr_and_ssim(image1, image2, border_size=0):
    """
    Computes PSNR and SSIM index from 2 images.
    We round it and clip to 0 - 255. Then shave 'scale' pixels from each border.
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)
    
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None
    
    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)
       
    return psnr, ssim


def psnr1(im1, im2):
    diff =np.float64(im1[:]) - np.float64(im2[:])
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20 * np.log10(255/rmse)
    return psnr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim1(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


from scipy.signal import convolve2d


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def norm(x):
    # input any range
    # Normalized to 0~1
    max = np.max(x)
    # if max < 1e-4:
    #     max = 0.0
    min = np.min(x)
    if max == min:
        normx = x
    else:
        normx = (x - min) / (max - min)
    return normx


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    
    out = np.zeros(arr.shape[1:] + (3,))
    
    eps = 1.e-20
    if pmin >= 0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0
    
    if pmax >= 0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1. + eps
    
    arr_norm = (1. * arr - mi) / (ma - mi + eps)
    
    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]
    
    return np.clip(out, 0, 1)


def savecolorim(save, im, norm=True, **imshow_kwargs):
    # im: Uint8
    imshow_kwargs['cmap'] = 'magma'
    if not norm:  # 不对当前图片归一化处理，直接保存
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 255
    
    im = np.asarray(im)
    im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    im = np.max(im, axis=proj_axis)

    # plt.imshow(im, **imshow_kwargs)
    # cb = plt.colorbar(fraction=0.05, pad=0.05)
    # cb.ax.tick_params(labelsize=23)  # 设置色标刻度字体大小。
    # # font = {'size': 16}
    # # cb.set_label('colorbar_title', fontdict=font)
    # plt.show()
    if save is not None:
        plt.imsave(save, im, **imshow_kwargs)
    else:
        # Make a random plot...
        fig = plt.figure()
        fig.add_subplot(111)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        plt.imshow(im, **imshow_kwargs)
        plt.axis('off')
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)

        # Now we can save it to a numpy array.
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


def savecolorim1(save, im, **imshow_kwargs):
    # save 方差 另一种颜色
    imshow_kwargs['cmap'] = 'cividis'
    
    im = np.asarray(im)
    im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    im = np.max(im, axis=proj_axis)
    plt.rc('font', family='Times New Roman')
    
    # plt.imshow(im, **imshow_kwargs)
    # cb = plt.colorbar(fraction=0.05, pad=0.05)
    # cb.ax.tick_params(labelsize=23)  # 设置色标刻度字体大小。
    # # font = {'size': 16}
    # # cb.set_label('colorbar_title', fontdict=font)
    # plt.show()
    
    plt.imsave(save, im, **imshow_kwargs)


import cv2
from PIL import Image


def savecolor_CV(save, y):
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(y, alpha=15), cv2.COLORMAP_JET)
    im = Image.fromarray(im_color)
    im.save(save)
