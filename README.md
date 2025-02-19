# OKD-CL

This is the repository for the OKD-CL dataset, <em>``Enhancing Object Recognition: The Role of Object Knowledge Decomposition and Component-Labeled Datasets''</em>.

Sample image distribution among instance categories within OKD-CL.
<div style="display:inline-block" align=center>
  <img src="https://github.com/XiGuaBo/OKD-CL/blob/master/ds_detail/instance_distribution_statisc.png" width=800 height=500>
</div>

A comparison of data metrics between the original dataset (PartImageNet) and the current one (OKD-CL). The fourth column provides a detailed breakdown of component categories and their respective counts for each primary category.
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

A statistics of OKD-CL natural language descriptions.
| General Attribute | Special Attribute | RelationShip Attribute |
|:---:|:---:|:---:|
| Shape, Color, Quantity (1,785) | 595 | 678 |

A statistics of OKD-CL split for train and validation.
| Train | Validation | Total |
|:---:|:---:|:---:|
| 12,625 | 1,489 | 14,114 |

## Ddtail of Method
***Activation Map Guidance:*** For any black box model, we define a feature extraction layer to the input as $A^k=f^l(x)$. After that, we obtain the activation map of this layer based on the Grad-Cam method, $L_{Grad-Cam}^c = ReLU(\sum_{k}{a_{k}^c \cdot A^k})$, where $a_{k}^c$ is the gradient contribution of the feature map of channel $k$ to a specific category $c$. From this we can define an activation map guided loss for three different masks as $L_{act}=MSE(M,L_{Grad-Cam}^c)$, where $M\in\lbrace0,1\rbrace^{h,w}$ is the corresponding mask downscale to $A^k$. 

***Knowledge Injection:*** Similar to the activation map constraint, we define the function of any hidden layer between the feature extraction layer and the previous logits layer for the input as $V=g^l(A)$. We embed class-specific component properties and relationships in OKD-CL as $V_{k}^c$. From this, we align $V_{k}^c$ and $V$ output by the model using MSE, $L_{ki}=MSE(V_{k}^c,V)$.

## Download the dataset

You can download the dataset from the github follow this [link to download dataset](https://github.com/XiGuaBo/OKD-CL/releases/tag/DATASET) and download the model weights follow this [link to download weights from Git](https://github.com/XiGuaBo/OKD-CL/releases/tag/WEIGHTS). 

The dataset should contain directories for <em>''train, test, hard_masks, instance-level_masks & instance-level_masks_visable and component-level_masks & component-level_masks_visable''</em>. (The knowledge vectors had been restored in code source as 99 pt files of the knowledge directory). Train and test directory both have 99 sub-directories of 99 object categories' image instances and directories of masks also have the same sub-directories structure including corresponding pixel-level annotation. 

Note that we assume the researcher has access to Hard-ImageNet and PartImageNet; the files above only contain the data of OKD-CL (image instances,3 kinds of masks and knowledge vectors in the code source and corresponding visable masks) we collected. You can download Hard-ImageNet and PartImageNet follow this [link to download Hard-ImageNet from box](https://umd.box.com/s/gx5qx4w03dgsumjclo7wpbdqov4xxrly) and [link to download PartImageNet from Google Drive](https://drive.google.com/file/d/1rZAECl3XF55NqJfW7Z9N63MFAvXupNuy/view?pli=1). 

## Setting up the data

1. Donwloads the dataset files and unzip them locally.
2. Downloads the checkpoint files and unzip them locally. 
3. Modified the dataset dirs in utils/para.py.

## Evaluate Models

We provide training codes for all 7 models, corresponding parameter settings and corresponding model weights can be downloaded in realese/Weights-CheckPoint.tar.gzxx include (VGG16, IncptionV1, MobileNetV1, MobileNetV2, ResNet50, ResNet101, ResNet152). We provide codes for all 3 evaluation methods including (ACC & FRR metrics evaluation, masking & noise perturbation test, method migration evaluation).

1. ACC & FRR metrics evaluation is in the train_modelname.py intergrate as a function named Metrics_Test with the model training code.
2. masking & noise perturbation test is in the disturbance.py (*related parameters defined in utiles/para.py).
3. method migration evaluation is in a independent sub-directory named "method_transfer" in the same level directory with other evaluation code (This sub-directory has a similar file structure to the parent directory).

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
