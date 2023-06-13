# OKD-CL

This is the repository for the OKD-CL dataset, associated with the NeurIPS 2023 Datasets and Benchmarks submission, <em>``Object Knowledge Decomposition and Components Label Dataset for Knowledge Guided Object Recognition''</em>.

## Download the dataset

You can download the dataset directly follow this [link to download from github
([https://github.com/XiGuaBo/OKD-CL/archive/refs/tags/v0.0.1.tar.gz]). 

The dataset should contain directories for <em>''train ,test ,hard_masks ,instance-level_masks and component-level_masks''</em>. (The knowledge vectors had been restored in code source as 99 txt files of the knowledge directory). Train and test directory both have 99 sub-directories of 99 object categories' image instances and directories of masks also have the same sub-directories structure including corresponding pixel-level annotation. 

Note that we assume the researcher has access to Hard-ImageNet; the files above only contain the data of OKD-CL (image instances,3 kinds of masks and knowledge vectors in the code source) we collected. You can download Hard-ImageNet follow this [link to download from box
(https://umd.box.com/s/gx5qx4w03dgsumjclo7wpbdqov4xxrly). 

## Setting up the data

1. Donwloads the dataset files and unzip them locally.
2. Downloads the checkpoint files and unzip them locally. 
3. Modified the dataset dirs in dataset/utiles.py.

## Evaluate Models

We provide training codes for all 7 models, corresponding parameter settings and corresponding model weights can be downloaded in realese/SaveWeights.tar.gz include (VGG16, IncptionV1, MobileNetV1, MobileNetV2, ResNet50, ResNet101, ResNet152). We provide codes for all 3 evaluation methods including (ACC & FRR metrics evaluation, masking & noise perturbation test, method migration evaluation).

1. ACC & FRR metrics evaluation is in the train_modelname.py intergrate as a function named Metrics_Test with the model training code.
2. masking & noise perturbation test is in the disturbance.py(*related parameters defined in utiles/para.py).
3. method migration evaluation is in a independent sub-directory named Hard-ImageNet Knowledge Guidance in the same level directory with other evaluation code(This sub-directory has a similar file structure to the parent directory).

## Environment

name: revrx
channels:
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_kmp_llvm
  - ca-certificates=2022.12.7=ha878542_0
  - certifi=2021.5.30=py36h5fab9bb_0
  - cudatoolkit=11.6.0=hecad31d_11
  - cudnn=8.4.0.27=hed8a83a_1
  - ld_impl_linux-64=2.38=h1181459_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=12.2.0=h65d4601_19
  - libstdcxx-ng=12.2.0=h46fd767_19
  - libzlib=1.2.13=h166bdaf_4
  - llvm-openmp=15.0.7=h0cdce71_0
  - ncurses=6.4=h6a678d5_0
  - openssl=1.1.1t=h0b41bf4_0
  - python=3.6.13=h12debd9_1
  - python_abi=3.6=2_cp36m
  - readline=8.2=h5eee18b_0
  - sqlite=3.40.1=h5082296_0
  - tk=8.6.12=h1ccaba5_0
  - xz=5.2.10=h5eee18b_1
  - zlib=1.2.13=h166bdaf_4
  - pip:
    - absl-py==0.15.0
    - aiohttp==3.8.4
    - aiosignal==1.2.0
    - astunparse==1.6.3
    - async-timeout==4.0.2
    - asynctest==0.13.0
    - attrs==22.2.0
    - cached-property==1.5.2
    - cachetools==4.2.4
    - charset-normalizer==2.0.12
    - clang==5.0
    - click==8.0.4
    - cycler==0.11.0
    - dataclasses==0.8
    - datasets==2.4.0
    - decorator==4.4.2
    - dill==0.3.4
    - et-xmlfile==1.1.0
    - filelock==3.4.1
    - flatbuffers==1.12
    - frozenlist==1.2.0
    - fsspec==2022.1.0
    - gast==0.4.0
    - google-auth==2.16.2
    - google-auth-oauthlib==0.4.6
    - google-pasta==0.2.0
    - grpcio==1.48.2
    - h5py==3.1.0
    - huggingface-hub==0.4.0
    - idna==3.4
    - idna-ssl==1.1.0
    - imageio==2.15.0
    - imgviz==1.7.2
    - importlib-metadata==4.8.3
    - importlib-resources==5.4.0
    - joblib==1.1.1
    - keras==2.6.0
    - keras-preprocessing==1.1.2
    - kiwisolver==1.3.1
    - labelme==5.1.1
    - markdown==3.3.7
    - matplotlib==3.3.4
    - multidict==5.2.0
    - multiprocess==0.70.12.2
    - natsort==8.2.0
    - networkx==2.5.1
    - numpy==1.19.5
    - oauthlib==3.2.2
    - opencv-contrib-python==4.1.1.26
    - openpyxl==3.1.2
    - opt-einsum==3.3.0
    - packaging==21.3
    - pandas==1.1.5
    - pillow==8.4.0
    - pip==21.3.1
    - protobuf==3.19.6
    - pyarrow==6.0.1
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pyparsing==3.0.9
    - pyqt5==5.15.6
    - pyqt5-qt5==5.15.2
    - pyqt5-sip==12.9.1
    - python-dateutil==2.8.2
    - pytz==2023.3
    - pywavelets==1.1.1
    - pyyaml==6.0
    - qtpy==2.0.1
    - regex==2023.5.5
    - requests==2.27.1
    - requests-oauthlib==1.3.1
    - responses==0.17.0
    - rsa==4.9
    - sacremoses==0.0.53
    - scikit-image==0.17.2
    - scikit-learn==0.24.2
    - scipy==1.5.4
    - setuptools==59.6.0
    - six==1.15.0
    - tcav==0.2.2
    - tensorboard==2.10.1
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.1
    - tensorflow-estimator==2.6.0
    - tensorflow-gpu==2.6.0
    - tensorflow-hub==0.13.0
    - termcolor==1.1.0
    - threadpoolctl==3.1.0
    - tifffile==2020.9.3
    - tokenizers==0.12.1
    - torch==1.10.1
    - torchaudio==0.10.1
    - torchvision==0.11.2
    - tqdm==4.64.1
    - transformers==4.18.0
    - typing-extensions==3.7.4.3
    - urllib3==1.26.15
    - werkzeug==2.0.3
    - wheel==0.37.1
    - wrapt==1.12.1
    - xxhash==3.2.0
    - yarl==1.7.2
    - zipp==3.6.0

## HardWare Requirement

NVIDIA GTX3090 24G x1
NVIDIA Driver Release 510.47.03
CUDA_TOOL_KIT Release 11.6

<!-- ## Citation -->

<!-- If the dataset or code is of use to you, please consider citing: -->
