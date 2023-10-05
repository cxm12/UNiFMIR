# UniFMIR
Pre-training a Foundation Model for Universal Fluorescence Microscopy Image Restoration

## Online Demo

We provide a live demo for UniFMIR at [http://unifmir.fdudml.cn/](http://unifmir.fdudml.cn/). If you only want to run the UniFMIR model, please refer to the [tutorial](https://github.com/cxm12/UNiFMIR#readme).

![demo](./demo.jpg)

## Training from Scratch

We provide the training code for UniFMIR. You can train the UniFMIR model from scratch by the following steps. 

### 1. Download Datasets

Please download preprocessed datasets from [the zenodo repository](https://doi.org/10.5281/zenodo.8401470). Or you can download the datasets from the original repository and preprocess the dataset according to the following literatures:

* The 3D denoising/isotropic reconstruction/projection datasets can be downloaded from [Content Aware Image Restoration dataset](https://publications.mpi-cbg.de/publications-sites/7207/). `Projection_Flywing/train_data/my_training_data.npz` are generated according to the [CSBDeep](http://csbdeep.bioimagecomputing.com/doc/). You can refer to the `Flywing_datagen.py` in `Preprocess` folder for details.

* The SR dataset can be downloaded from [BioSR dataset](https://doi.org/10.6084/m9.figshare.13264793). Please augment the dataset according to the instructions in [DFCAN](https://github.com/qc17-THU/DL-SR/tree/main#train-a-new-model) and then generate `my_training_data.npz` files following [CSBDeep](http://csbdeep.bioimagecomputing.com/doc/datagen.html). You can refer to the `BioSR_DataAugumentation.m` and `BioSR_datagen.py` in `Preprocess` folder for details.

* The Volumetric reconstruction dataset are from [VCD-LFM dataset](https://doi.org/10.5281/zenodo.4390067). Please prepare the dataset according to the instructions in [VCD-Net](https://github.com/feilab-hust/VCD-Net).

* DeepBacs dataset can be downloaded from [DeepBacs dataset](https://zenodo.org/record/6460867). We split the dataset into 5 folds for cross-validation. Shareloc dataset can be downloaded from [Shareloc dataset](https://zenodo.org/record/7234161).

The data path should be as follows:

```
VCD/vcdnet/
CSB/DataSet/
    Denoising_Planaria/
    Denoising_Tribolium/
    Isotropic/Isotropic_Liver/
    Projection_Flywing/
    BioSR_WF_to_SIM/DL-SR-main/dataset/
    Synthetic_tubulin_gfp/
    Synthetic_tubulin_granules/
DeepBacs/
Shareloc/
```

### 2. Pretrain the UniFMIR model

We provide the pre-training code for UniFMIR. You can pre-train the UniFMIR model by employing the following steps. You can also download the pretrained UniFMIR model from [the release](https://github.com/cxm12/UNiFMIR/releases).

* Modify `mainUni.py` 

Replacing "srdatapath", "denoisedatapath", "isodatapath", "prodatapath"， "voldatapath" with the folder name containing training images for different tasks.

* Run the `mainUni_pretrain.py` script and save the model in the folder `experiment/Uni-SwinIR/`.

```
python mainUni_pretrain.py
```

### 3. Finetune the UniFMIR model

We provide the finetuning code for UniFMIR. You can finetune the pretrained UniFMIR model by the following steps. The pretrained model from the above step should be put in the path `experiment/Uni-SwinIR/model/model_best.pt`.

* Modify `mainUni.py` 
  
```
task = 1 # SR: 1, Denoising: 2, Isotropic: 3, Projection: 4, Volumetric: 5
test_only = False
pretrain = './experiment/Uni-SwinIR/model/model_best.pt'
```

Replacing "srdatapath", "denoisedatapath", "isodatapath", "prodatapath"， "voldatapath" with the folder name containing training images for different tasks.

* Run the `mainUni.py` script and save the model in the folder `experiment/Uni-SwinIR[dataset]/`.

```
python mainUni.py
```

* Modify `mainUni.py` 
  
```
test_only = True
pretrain = './experiment/Uni-SwinIR[dataset]/model/model_best.pt'
```

* Run the `mainUni.py` script and show the test result.

```
python mainUni.py
```

## Finetuning for DeepBacs dataset

We provide the finetuning code for DeepBacs dataset. You can finetune the pretrained UniFMIR model by the following steps. The pretrained model from the above step should be put in the path `experiment/Uni-SwinIR/model/model_best.pt`.

* Modify `mainDeepBacs2.py` 
  
```
task = 1 # SR: 1, Denoising: 2
test_only = False
pretrain = './experiment/Uni-SwinIR/model/model_best.pt'
```

Replacing "traindatapath", with the folder name containing training images for DeepBacs.

* Run the `mainDeepBacs2.py` script and save the model in the folder `experiment/Uni-SwinIR[dataset]/`.

```
python mainDeepBacs2.py
```

* Modify `mainDeepBacs2.py` 

```
test_only = True
```

* Run the `mainDeepBacs2.py` script and show the test result.

```
python mainDeepBacs2.py
```

## Finetuning for Shareloc dataset

We provide the finetuning code for Shareloc dataset. You can finetune the pretrained UniFMIR model by the following steps. The pretrained model from the above step should be put in the path `experiment/Uni-SwinIR/model/model_best.pt`.

* Modify `mainShareloc2.py` 
  
```
task = 1
test_only = False
pretrain = './experiment/Uni-SwinIR/model/model_best.pt'
```

Replacing "traindatapath", with the folder name containing training images for Shareloc.

* Run the `mainShareloc2.py` script and save the model in the folder `experiment/Uni-SwinIR[dataset]/`.

```
python mainShareloc2.py
```

* Modify `mainShareloc2.py` 
  
```
test_only = True
pretrain = './experiment/Uni-SwinIR[dataset]/model/model_best.pt'
```

* Run the `mainShareloc2.py` script and show the test result.

```
python mainUni.py
```