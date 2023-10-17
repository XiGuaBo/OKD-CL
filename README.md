# OKD-CL

This is the repository for the OKD-CL dataset, associated with the CVPR 2024 submission, <em>``Object Knowledge Decomposition and Components Label Dataset for Knowledge Guided Object Recognition''</em>.

<div style="display:inline-block" align=center>
  <img src="https://github.com/XiGuaBo/OKD-CL/blob/master/ds_detail/instance_distribution_statisc.png" width=800 height=500>
</div>

<div style="display:inline-block" align=center>
| Supper-Class | Class Nums (Origin) | Class Nums (Filtered) | Annotated Parts |
|:---:|:---:|:---:|:---:|
| Quadruped | 46 | 35 | Head, Body, Foot, Tail |
| Biped | 17 | 8 | Head, Body, Hand, Foot, Tail |
| Fish | 10 | 7 | Head, Body, Fin, Tail |
| Bird | 14 | 9 | Head, Body, Wing, Foot, Tail |
| Snake | 15 | 4 | Head, Body |
| Reptile | 20 | 8 | Head, Body, Foot, Tail |
| Car | 20 | 14 | Body, Tier, Side Mirror |
| Bicycle | 5 | 5 |Head, Body, Seat, Tier |
| Boat | 4 | 4 | Body, Sail |
| Aeroplane | 2 | 2 | Head, Body, Wing, Engine, Tail |
| Bottle | 5 | 3 | Body, Mouth |
| Total | 158 | 99 | / |
</div>

## Download the dataset

You can download the dataset directly follow this [link to download from github
(https://github.com/XiGuaBo/OKD-CL/archive/refs/tags/v0.0.1.tar.gz). 

The dataset should contain directories for <em>''train ,test ,hard_masks ,instance-level_masks and component-level_masks''</em>. (The knowledge vectors had been restored in code source as 99 pt files of the knowledge directory). Train and test directory both have 99 sub-directories of 99 object categories' image instances and directories of masks also have the same sub-directories structure including corresponding pixel-level annotation. 

Note that we assume the researcher has access to Hard-ImageNet; the files above only contain the data of OKD-CL (image instances,3 kinds of masks and knowledge vectors in the code source) we collected. You can download Hard-ImageNet follow this [link to download from box
(https://umd.box.com/s/gx5qx4w03dgsumjclo7wpbdqov4xxrly). 

## Setting up the data

1. Donwloads the dataset files and unzip them locally.
2. Downloads the checkpoint files and unzip them locally. 
3. Modified the dataset dirs in utils/para.py.

## Evaluate Models

We provide training codes for all 7 models, corresponding parameter settings and corresponding model weights can be downloaded in realese/SaveWeights.tar.gz include (VGG16, IncptionV1, MobileNetV1, MobileNetV2, ResNet50, ResNet101, ResNet152). We provide codes for all 3 evaluation methods including (ACC & FRR metrics evaluation, masking & noise perturbation test, method migration evaluation).

1. ACC & FRR metrics evaluation is in the train_modelname.py intergrate as a function named Metrics_Test with the model training code.
2. masking & noise perturbation test is in the disturbance.py(*related parameters defined in utiles/para.py).
3. method migration evaluation is in a independent sub-directory named Hard-ImageNet Knowledge Guidance in the same level directory with other evaluation code(This sub-directory has a similar file structure to the parent directory).

## Dataset Detail

1. We have uploaded the annotated textual documents (xls) for component categorization and component attributes of OKD-CL and Hard-ImageNet.

## Environment

- keras==2.6.0
- matplotlib==3.3.4
- numpy==1.19.5
- openai==0.8.0
- opencv_contrib_python==4.1.1.26
- openpyxl==3.1.2
- scikit_learn==1.3.1
- tensorflow==2.14.0
- tensorflow_gpu==2.6.0
- torch==1.10.1
- tqdm==4.64.1

## HardWare Requirement

- NVIDIA GTX3090 24G x1
- NVIDIA Driver Release 510.47.03
- CUDA_TOOL_KIT Release 11.6

<!-- ## Citation -->

<!-- If the dataset or code is of use to you, please consider citing: -->
