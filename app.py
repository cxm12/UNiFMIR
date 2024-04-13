import gradio as gr
import numpy as np
import random
import imageio
from tifffile import imread, imsave

import os, cv2
from tqdm import tqdm
import torch
import utility
import model
from div2k import normalize, PercentileNormalizer

DEVICES = ['CPU','CUDA','Paralleled CUDA']
QUANT = ['float32','float16',]
TASKS = ['SR_Microtubules','SR_CCPs','SR_F-actin','SR_ER','Isotropic_Liver','Projection_Flywing','Denoising_Planaria','Denoising_Tribolium','Volumetric_VCD']
INPUTS = ['SR', 'Denoising', 'Isotropic', 'Projection', 'Volumetric']
MODEL = None
ARGS = None

class Args:
    model = 'SwinIR'
    test_only = True
    resume = 0
    modelpath = None
    save = None
    task = None
    dir_data = None
    dir_demo = None
    data_test = None

    epoch = 1000
    batch_size = 16
    patch_size = None
    rgb_range = 1
    n_colors = 1
    inch = None
    datamin = 0
    datamax = 100
    
    cpu = False
    print_every = 1000
    test_every = 2000
    load=''
    lr = 0.00005
    n_GPUs = 1
    n_resblocks = 8
    n_feats = 32
    save_models = True
    save_results = True
    save_gt = False

    debug = False
    scale = None
    chunk_size = 144
    n_hashes = 4
    chop = False
    self_ensemble = False
    no_augment = False
    inputchannel = None

    act = 'relu'
    extend = '.'
    res_scale = 0.1
    shift_mean = True
    dilation = False
    precision = 'single'

    seed = 1
    local_rank = 0
    n_threads = 0
    reset = False
    split_batch = 1
    gan_k = 1

def load_model(type, device, chop, quantization, skip, progress=gr.Progress()):
    global MODEL, ARGS

    ARGS = Args()

    if quantization == 'float16':
        ARGS.precision = 'half'

    if chop == 'Yes':
        ARGS.chop = True

    if device == 'CPU':
        ARGS.cpu = True
    elif device == 'CUDA':
        ARGS.cpu = False
        ARGS.n_GPUs = 1
    elif device == 'Paralleled CUDA':
        ARGS.cpu = False
        ARGS.n_GPUs = torch.cuda.device_count()
    else:
        gr.Error("Device not found!")
        return "Device not found"

    if 'SR' in type:
        ARGS.task = 1
        ARGS.patch_size = 128
        ARGS.scale = '2'
        ARGS.inch = 1
        # ARGS.chop = False

        if type == 'SR_F-actin':
            ARGS.save = 'SwinIRF-actin'
            ARGS.modelpath = './experiment/SwinIRF-actin/model_best181.pt'
        elif type == 'SR_CCPs':
            ARGS.save = 'SwinIRCCPs'
            ARGS.modelpath = './experiment/SwinIRCCPs/model_best.pt'
        elif type == 'SR_ER':
            ARGS.save = 'SwinIRER'
            ARGS.modelpath = './experiment/SwinIRER/model_best147.pt'
        elif type == 'SR_Microtubules':
            ARGS.save = 'SwinIRMicrotubules'
            ARGS.modelpath = './experiment/SwinIRMicrotubules/model_best.pt'
        else:
            gr.Error("Model not found!")
            return "Model not found"
        

    elif 'Denoising' in type:
        ARGS.task = 2
        ARGS.patch_size = 64
        ARGS.scale = '1'
        
        if type == 'Denoising_Planaria':
            ARGS.model = 'SwinIR'
            ARGS.inputchannel = 1
            ARGS.resume = -15
            ARGS.save = 'SwinIRDenoising_Planaria'
            ARGS.modelpath = './experiment/SwinIRDenoising_Planaria/model_best.pt'
        elif type == 'Denoising_Tribolium':
            ARGS.model = 'SwinIRmto1'
            ARGS.inputchannel = 5
            ARGS.resume = 0
            ARGS.save = 'SwinIRmto1Denoising_Tribolium'
            ARGS.modelpath = './experiment/SwinIRmto1Denoising_Tribolium/model_best.pt'
        else:
            gr.Error("Model not found!")
            return "Model not found"

    elif 'Isotropic' in type:
        ARGS.task = 3
        ARGS.patch_size = 128
        ARGS.scale = '1'

        ARGS.model = 'SwinIR'
        ARGS.save = 'SwinIRIsotropic_Liver'
        ARGS.resume = 0
        ARGS.modelpath = './experiment/SwinIRIsotropic_Liver/model_best465.pt'

    elif 'Projection' in type:
        ARGS.task = 4
        ARGS.patch_size = 128
        ARGS.scale = '1'
        ARGS.inch = 50

        ARGS.model = 'SwinIRproj2stg_enlcn_2npz'
        ARGS.save = 'SwinIRproj2stg_enlcn_2npzProjection_Flywing'
        ARGS.resume = -6
        ARGS.modelpath = './experiment/SwinIRproj2stg_enlcn_2npzProjection_Flywing/model_best6.pt'

    elif 'Volumetric' in type:
        ARGS.task = 5
        ARGS.patch_size = 176
        ARGS.scale = '1'

        ARGS.model = 'SwinIR2t3_stage2'
        ARGS.save = 'SwinIR2t3_stage2VCD'
        ARGS.resume = -17
        ARGS.modelpath = './experiment/SwinIR2t3_stage2VCD/model_best17.pt'
    else:
        gr.Error("Task not found!")
        return "Task not found"
    
    ARGS.scale = list(map(lambda x: int(x), ARGS.scale.split('+')))
    if MODEL is not None:
        del MODEL

    checkpoint = utility.checkpoint(ARGS)
    MODEL = model.Model(ARGS, checkpoint)
    MODEL.eval()

    if skip == 'Yes' and ARGS.n_GPUs <= 1:
        if 'Projection' in type:
            MODEL.model.denoise.layers[1].prune()
        else:
            MODEL.model.layers[1].prune()

    return '%s Model loaded on %s with %s precision'%(type, device, quantization)

def visualize(img_input, progress=gr.Progress()):
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif'):
        gr.Error("Image must be a tiff file!")
        return None
    
    image = imread(img_input.name)
    shape = image.shape
    print(f'Image shape: {shape}')

    if len(shape) == 2:
        image = utility.savecolorim(None, image, norm=True)
        return [[image], f'2D image loaded with shape {shape}']
    elif len(shape) == 3:
        clips = []
        for i in range(shape[0]):
            clips.append(utility.savecolorim(None, image[i], norm=True))
        return [clips, f'3D image loaded with shape {shape}']
    else:
        gr.Error("Image must be 2 or 3 dimensional!")
        return None
    
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """

    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:, :, d] = image[d, :, :]
    return image_re

def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                        False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))
    
        if mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra

def _load_imgs(img_file, t2d=True):
    def normalize(x):
        max_ = np.max(x) * 1.1
        x = x / (max_ / 2.)
        x = x - 1
        return x
    
    if t2d:
        image = imageio.imread(img_file)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # uint8 0~48 (176,176,1) (649, 649,1)
        img = normalize(image)  # float64 -1~1 (176,176,1)
        img = lf_extract_fn(img, n_num=11, padding=False)  # (16, 16, 121) (59, 59, 121)
    else:
        image = imageio.volread(img_file)  # uint8 0~132  [61,176,176]
        img = normalize(image)  # float64 -1~1 (61,176,176)
        img = rearrange3d_fn(img)  # (176,176,61)

    img = img.astype(np.float32, casting='unsafe')
    # print('\r%s : %s' % (img_file, str(img.shape)), end='')
    return img

def run_model_corr(img_input, type, corr, progress=gr.Progress()):
    img, axes = run_model(img_input, type)

    if img is None:
        return [None, None]
    
    if corr == 'Yes':
        img_rs, axes = run_model(img_input, type, resize=True)
        print("corr", img.shape, img_rs.shape)
        if len(img.shape) == 3:
            img_rs = img_rs.transpose(1, 2, 0)
        img_rs = cv2.resize(img_rs, (img.shape[-1], img.shape[-2]), interpolation=cv2.INTER_CUBIC)
        if len(img.shape) == 3:
            img_rs = img_rs.transpose(2, 0, 1)
        img = (img + img_rs) / 2

    utility.save_tiff_imagej_compatible('output.tif', img, axes)
    return ['output.tif', "Output Successfully Saved!"]

@torch.no_grad()
def run_model(img_input, type, resize=False):
    global MODEL, ARGS
    
    if MODEL is None:
        gr.Error("Model not loaded!")
        return [None, None]

    if img_input is None:
        gr.Error("Image not loaded!")
        return [None, None]  
    
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif'):
        gr.Error("Image must be a tiff file!")
        return [None, None]
    
    normalizer = PercentileNormalizer(2, 99.8)

    if 'SR' in type:
        image = imread(img_input.name)

        if image.ndim != 2:
            gr.Error("SR Image must be 2 dimensional!")
            return [None, None]
        
        # expand to 4 dimensions tensor
        lr = normalize(image, ARGS.datamin, ARGS.datamax, clip=True) * ARGS.rgb_range
        lr = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float()

        if resize:
            lr = torch.nn.functional.interpolate(lr, scale_factor=1/2, mode='bicubic', align_corners=True)

        # model inference
        sr = MODEL(lr.to(MODEL.device), 0)

        # normalize to 0-1
        sr = utility.quantize(sr, ARGS.rgb_range)

        # convert to numpy
        sr = sr.float().squeeze(0).squeeze(0).cpu().detach().numpy()

        # save image
        # imsave('output.tif', sr)

        # visualize
        # sr_norm = utility.savecolorim(None, sr, norm=True)
        return [sr, "YX"]

    elif 'Denoising' in type:
        print(f'Opening {img_input.name}...')
        image = imread(img_input.name)

        if image.ndim != 3:
            gr.Error("Denoising Image must be 3 dimensional!")
            return [None, None]

        # expand to 4 dimensions tensor
        lrt = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        lrt = normalizer.before(lrt, 'CZYX').to(MODEL.device)

        # model inference
        denoiseim = torch.zeros_like(lrt, dtype=lrt.dtype)
        if resize:
            denoiseim = torch.nn.functional.interpolate(denoiseim, scale_factor=1/2, mode='bicubic', align_corners=True)
        batchstep = ARGS.n_GPUs * 4
        inputlst = []
        for ch in range(0, lrt.shape[1]):  # [45, 486, 954]  0~44
            # print(ch)
            if ch < ARGS.inputchannel//2:  # 0, 1
                lr1 = [lrt[:, ch:ch+1, :, :] for _ in range(ARGS.inputchannel//2-ch)]
                lr1.append(lrt[:, :ARGS.inputchannel//2+ch+1])
                lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
            elif ch >= (lrt.shape[1] - ARGS.inputchannel//2):  # 43, 44
                lr1 = []
                lr1.append(lrt[:, ch-ARGS.inputchannel // 2:])
                numa = (ARGS.inputchannel // 2 - (lrt.shape[1] - ch)) + 1
                lr1.extend([lrt[:, ch:ch+1, :, :] for _ in range(numa)])
                lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
            else:
                lrt1 = lrt[:, ch-ARGS.inputchannel // 2:ch + ARGS.inputchannel // 2 + 1]
            assert lrt1.shape[1] == ARGS.inputchannel
            inputlst.append(lrt1)
                
        for dp in range(0, len(inputlst), batchstep):
            if dp + batchstep >= len(inputlst):
                dp = len(inputlst) - batchstep
            print(dp)  # 0, 10, .., 90
            lrtn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inputchannel, h, w]
            if resize:
                lrtn = torch.nn.functional.interpolate(lrtn, scale_factor=1/2, mode='bicubic', align_corners=True)

            a = MODEL(lrtn, 0)
            a = torch.transpose(a, 1, 0)  # [1, batch, h, w]
            denoiseim[:, dp:dp + batchstep, :, :] = a
        
        # normalize to 0-1
        sr = np.float32(denoiseim.cpu().detach().numpy())
        sr = np.squeeze(normalizer.after(sr))
        # imsave('output.tif', sr)
        
        # save image
        # sr_norm = np.squeeze(np.float32(normalize(sr, ARGS.datamin, ARGS.datamax, clip=True)))
        # clips = []
        # for i in range(sr_norm.shape[0]):
        #     clips.append(utility.savecolorim(None, sr_norm[i], norm=True))
        return [sr, "CYX"]
    
    elif 'Isotropic' in type:
        image = imread(img_input.name)

        if image.ndim != 3:
            gr.Error("Isotropic Image must be 3 dimensional!")
            return [None, None]

        # expand to 4 dimensions tensor
        lr = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        lr = normalizer.before(lr, 'CZYX')
        lr = np.float32(np.squeeze(lr.cpu().detach().numpy()))
        lr = np.expand_dims(lr, -1)

        def _rotate(arr, k=1, axis=1, copy=True):
            """Rotate by 90 degrees around the first 2 axes."""
            if copy:
                arr = arr.copy()
            k = k % 4
            arr = np.rollaxis(arr, axis, arr.ndim)
            if k == 0:
                res = arr
            elif k == 1:
                res = arr[::-1].swapaxes(0, 1)
            elif k == 2:
                res = arr[::-1, ::-1]
            else:
                res = arr.swapaxes(0, 1)[::-1]
        
            res = np.rollaxis(res, -1, axis)
            return res

        isoim1 = np.zeros_like(lr, dtype=np.float32)
        if resize:
            isoim1 = torch.nn.functional.interpolate(isoim1, scale_factor=1/2, mode='bicubic', align_corners=True)
        isoim2 = np.zeros_like(lr, dtype=np.float32)
        if resize:
            isoim2 = torch.nn.functional.interpolate(isoim2, scale_factor=1/2, mode='bicubic', align_corners=True)

        batchstep = ARGS.n_GPUs * 4
        for wp in tqdm(range(0, lr.shape[2], batchstep)):
            if wp + batchstep >= lr.shape[2]:
                wp = lr.shape[2] - batchstep
            # [d, h, w=batchstep, 1]-> [w=batchstep, h, d, 1]# [360,768,768,2] -> [768,768,360,2]
            x_rot1 = _rotate(lr[:, :, wp:wp + batchstep, :], axis=1, copy=False)
            # [w=batchstep, h, d, 1]-> [w=batchstep, h, d]-> [w=batchstep, 1, h, d]
            x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)

            x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()

            if resize:
                x_rot1 = torch.nn.functional.interpolate(x_rot1, scale_factor=1/2, mode='bicubic', align_corners=True)

            a1 = MODEL(x_rot1.to(MODEL.device), 0)

            # [w=batchstep, 1, h, d] -> [w=batchstep, h, d] -> [w=batchstep, h, d, 1]
            a1 = np.expand_dims(np.squeeze(a1.cpu().detach().numpy()), -1)
            # [w=batchstep, h, d, 1] -> [d, h, w=batchstep, 1]  # [360,768,768,2]
            u1 = _rotate(a1, -1, axis=1, copy=False)
            isoim1[:, :, wp:wp + batchstep, :] = u1

        for hp in tqdm(range(0, lr.shape[1], batchstep)):
            if hp + batchstep >= lr.shape[1]:
                hp = lr.shape[1] - batchstep

            # [d, h=batchstep, w, 1]-> [h=batchstep, w, d, 1] # [768,768,360,2]
            x_rot2 = _rotate(_rotate(lr[:, hp:hp + batchstep, :, :], axis=2, copy=False), axis=0, copy=False)
            # [h=batchstep, w, d, 1]-> [h=batchstep, w, d]-> [h=batchstep, 1, w, d]
            x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)

            x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()

            if resize:
                x_rot2 = torch.nn.functional.interpolate(x_rot2, scale_factor=1/2, mode='bicubic', align_corners=True)

            a2 = MODEL(x_rot2.to(MODEL.device), 0)

            # [h=batchstep, 1, w, d] -> [h=batchstep, w, d] -> [h=batchstep, w, d, 1]
            a2 = np.expand_dims(np.squeeze(a2.cpu().detach().numpy()), -1)
            # [h=batchstep, w, d, 1] -> [d, h=batchstep, w, 1]  # [360,768,768,2]
            u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
            isoim2[:, hp:hp + batchstep, :, :] = u2

        sr = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
        sr = np.float32(np.squeeze(normalizer.after(sr)))
        # imsave('output.tif', sr)

        # save image
        # sr_norm = np.squeeze(np.float32(normalize(sr, ARGS.datamin, ARGS.datamax, clip=True)))
        # clips = []
        # for i in range(sr_norm.shape[0]):
        #     clips.append(utility.savecolorim(None, sr_norm[i], norm=True))
        return [sr, "CYX"]
    
    elif 'Projection' in type:
        image = imread(img_input.name)

        if image.ndim != 3:
            gr.Error("Projection Image must be 3 dimensional!")
            return [None, None]

        # expand to 4 dimensions tensor
        lr = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        if resize:
            lr = torch.nn.functional.interpolate(lr, scale_factor=1/2, mode='bicubic', align_corners=True)

        a = MODEL(lr.to(MODEL.device), 0)
        sr = np.float32(np.squeeze(a[0].cpu().detach().numpy()))

        # save image
        # print(sr.shape)
        srtf = np.squeeze(sr)
        # axes_restored = 'YX'
        # utility.save_tiff_imagej_compatible('output.tif', srtf, axes_restored)

        # visualize
        # sr_norm = np.squeeze(np.float32(normalize(srtf, ARGS.datamin, ARGS.datamax, clip=True)))
        # sr_norm = utility.savecolorim(None, sr_norm, norm=True)
        return [srtf, "YX"]
    
    elif 'Volumetric' in type:
        image = imread(img_input.name)

        if image.ndim != 2:
            gr.Error("Volumetric Image must be 2 dimensional!")
            return [None, None]

        # load image
        lr = _load_imgs(img_input.name, True)
        lr = np.transpose(lr, (2, 0, 1))[None, ...]
        lr = torch.from_numpy(np.ascontiguousarray(lr * ARGS.rgb_range)).float()

        if resize:
            lr = torch.nn.functional.interpolate(lr, scale_factor=1/2, mode='bicubic', align_corners=True)

        # model inference
        a = MODEL(lr.to(MODEL.device), 0)
        sr = np.float32(a.cpu().detach().numpy())

        # save image
        sr_norm = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2 
        # imsave('output.tif', sr_norm)

        # # visualize
        # clips = []
        # for i in range(sr_norm.shape[0]):
        #     clips.append(utility.savecolorim(None, sr_norm[i], norm=True))
        return [sr_norm, "CYX"]

    else:
        gr.Error("This task is not supported yet!")
        return [None, None]

with gr.Blocks() as demo:

    gr.Markdown("# UniFMIR: Pre-training a Foundation Model for Universal Fluorescence Microscopy Image Restoration")
    gr.Markdown("This demo allows you to run the models on your own images or the examples  from the paper. Please refer to the paper for more details.")

    gr.Markdown("## Instructions")
    gr.Markdown("1. Upload your tiff image or use the examples below. We accept 2 (xy) dimensional images for SR and Volumetric Reconstruction and 3 (zxy) dimensional images for Denoising, Projection and Isotropic Reconstruction.")
    gr.Markdown("2. Click 'Check Input' to inspect your input image. This may take a while to display the image.")
    gr.Markdown("3. Select the model you want to run. We provide models for different tasks and datasets, including SR (CCPs, ER, Microtubules, F-actin), Denoising (Planaria, Tribolium),Isotropic (Liver), Projection (Flywing), Volumetric (VCD).")
    gr.Markdown("3. Select the device and quantization you want to run the model on. We support CPU, GPU, and multiple GPUs. Float16 will save time and memory with almost no performance drop.")
    gr.Markdown("5. Select the model options. You can choose to chop the image into smaller patches to save memory. Pixel size correction will take longer to run but may produce better results with large input resolution. Fast inference will skip one Swin block to accelerate but may result in some performance drop.")
    gr.Markdown("6. Click 'Load Model' to load the model. This may take a while.")
    gr.Markdown("7. Click 'Restore Image' to run the model on the input image. Some tasks like denoising will take several minutes to run. The output image will be saved as 'output.tif' for download.")
    gr.Markdown("8. Click 'Check Output' to inspect the output image. This may take a while to display the image.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Image or Use Examples")
                
            with gr.Column():
                img_input = gr.File(label="Input File", interactive=True)
                img_visual = gr.Gallery(label="Input Viusalization", interactive=False)

            with gr.Row():
                input_message = gr.Textbox(label="Image Information", value="Image not loaded")
                check_input = gr.Button("Check Input") 

            with gr.Row():
                with gr.Column():
                    gr.Examples(
                        label='Super Resolution Examples',
                        examples=[
                            ["exampledata/BioSR/Microtubes.tif",'SR'],
                            ["exampledata/BioSR/CCP.tif",'SR'],
                            ["exampledata/BioSR/F-actin.tif",'SR'],
                        ],
                        inputs=[img_input, input_message],
                    )

                    gr.Examples(
                        label='Isotropic Examples',
                        examples=[
                            ["exampledata/Isotropic/Liver.tif",'Isotropic'],
                        ],
                        inputs=[img_input, input_message],
                    )

                    gr.Examples(
                        label='Projection Examples',
                        examples=[
                            ["exampledata/Proj/Flywing.tif","Projection"],
                        ],
                        inputs=[img_input, input_message],
                    )

                with gr.Column():
                    gr.Examples(
                        label='Denoising Examples',
                        examples=[
                            ["exampledata/Denoise/P/C1/Planaria_C1.tif","Denoising"],
                            ["exampledata/Denoise/P/C2/Planaria_C2.tif","Denoising"],
                            ["exampledata/Denoise/P/C3/Planaria_C3.tif","Denoising"],
                            ["exampledata/Denoise/T/C1/Tribolium_C1.tif","Denoising"],
                            ["exampledata/Denoise/T/C2/Tribolium_C2.tif","Denoising"],
                            ["exampledata/Denoise/T/C3/Tribolium_C3.tif","Denoising"],
                        ],
                        inputs=[img_input, input_message],
                    )

                    gr.Examples(
                        label='Volumetric Reconstruction Examples',
                        examples=[
                            ["exampledata/volumetricRec/VCD.tif","Volumetric"],
                        ],
                        inputs=[img_input, input_message],
                    )

        with gr.Column():
            gr.Markdown("## Load and Run Model")
            output_file = gr.File(label="Output File", interactive=False)
            img_output = gr.Gallery(label="Output Visualiztion")

            with gr.Row():
                type = gr.Dropdown(label="Model Type", choices=TASKS, value="SR_Microtubules")
                device = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA")
                quantization = gr.Dropdown(label="Quantization", choices=QUANT, value="float16")

            with gr.Row():
                chop = gr.Dropdown(label="Chop", choices=['Yes','No'], value="Yes")
                corr = gr.Dropdown(label="Pixel Size Correction", choices=['Yes','No'], value="No")
                skip = gr.Dropdown(label="Fast Inference", choices=['Yes','No'], value="No")

            with gr.Row():
                load_progress = gr.Textbox(label="Model Information", value="Model not loaded")
                load_btn = gr.Button("Load Model")
                run_btn = gr.Button("Restore Image")
                
            with gr.Row():
                output_message = gr.Textbox(label="Output Information", value="Image not loaded")
                display_btn = gr.Button("Check Output")

    check_input.click(visualize, inputs=img_input, outputs=[img_visual, input_message], queue=True)
    display_btn.click(visualize, inputs=output_file, outputs=[img_output, output_message], queue=True)
    load_btn.click(load_model,inputs=[type, device, chop, quantization, skip],outputs=load_progress, queue=True)
    run_btn.click(run_model_corr, inputs=[img_input, type, corr], outputs=[output_file, output_message], queue=True)

demo.queue().launch(server_name='0.0.0.0')
