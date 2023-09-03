# UniFMIR
"Pre-training a Foundation Model for Universal Fluorescence Microscopymage Restoration"

### Requirements
* Python 3.7
* CUDA 11.4 and CUDNN 
* Packages: 
  
  basicsr          ==          1.4.2
  easydict         ==          1.11.dev0
  imageio          ==          2.13.3
  keras            ==          2.11.0
  numpy            ==          1.21.5
  opencv-python    ==          4.5.4.60
  Pillow           ==          9.0.1
  scikit-image     ==          0.19.2
  scipy            ==          1.7.3
  tensorflow-gpu   ==          2.7.0
  tifffile         ==          2021.11.2
  torch            ==          1.10.0+cu113
  
csbdeep [![PyPI version](https://badge.fury.io/py/csbdeep.svg)](https://pypi.org/project/csbdeep)

### Pretrain the UniFMIR model

```
cd <directory of the .py file>
python mainUni_pretrain.py
```
Replacing "srdatapath", "denoisedatapath", "isodatapath", "prodatapath"， "voldatapath" with the folder name containing pre-training images for different tasks.


### Finetune the UniFMIR model

#### 3D denoising

```
cd <directory of the .py file>
python mainUni.py
```
* Modify `mainUni.py` 
  
  task = 2
  test_only = False
  
  Replacing "srdatapath", "denoisedatapath", "isodatapath", "prodatapath"， "voldatapath" with the folder name containing 
  pre-training images for different tasks.
  
  
### Prediction
Test the UniFMIR model on the 3D denoising task

```
cd <directory of the .py file>
python mainUni.py
```
Replacing "rootdatapath" with the folder name containing pre-training images.
  


### Data
All training and test data involved in the experiments are publicly available datasets. 

* The 3D denoising/isotropic reconstruction/projection datasets can be downloaded from `https://publications.mpi-cbg.de/publications-sites/7207/`

* The SR dataset can be downloaded from `https://doi.org/10.6084/m9.figshare.13264793`

* The Volumetric reconstruction dataset can be downloaded from `https://doi.org/10.5281/zenodo.4390067`

### Model
The pretrained models can be downloaded from `https://pan.baidu.com/`

