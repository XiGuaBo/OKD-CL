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
3. Modified the dataset dirs in utils/para.py.

## Evaluate Models

We provide training codes for all 7 models, corresponding parameter settings and corresponding model weights can be downloaded in realese/SaveWeights.tar.gz include (VGG16, IncptionV1, MobileNetV1, MobileNetV2, ResNet50, ResNet101, ResNet152). We provide codes for all 3 evaluation methods including (ACC & FRR metrics evaluation, masking & noise perturbation test, method migration evaluation).

1. ACC & FRR metrics evaluation is in the train_modelname.py intergrate as a function named Metrics_Test with the model training code.
2. masking & noise perturbation test is in the disturbance.py(*related parameters defined in utiles/para.py).
3. method migration evaluation is in a independent sub-directory named Hard-ImageNet Knowledge Guidance in the same level directory with other evaluation code(This sub-directory has a similar file structure to the parent directory).

## Environment

dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - _tflow_select=2.3.0=mkl
  - abseil-cpp=20211102.0=hd4dd3e8_0
  - absl-py=1.3.0=py39h06a4308_0
  - aiohttp=3.8.1=py39h7f8727e_1
  - aiosignal=1.2.0=pyhd3eb1b0_0
  - astunparse=1.6.3=py_0
  - async-timeout=4.0.2=py39h06a4308_0
  - attrs=21.4.0=pyhd3eb1b0_0
  - blas=1.0=openblas
  - blinker=1.4=py39h06a4308_0
  - brotlipy=0.7.0=py39h27cfd23_1003
  - c-ares=1.18.1=h7f8727e_0
  - ca-certificates=2022.07.19=h06a4308_0
  - cachetools=4.2.2=pyhd3eb1b0_0
  - certifi=2022.9.24=py39h06a4308_0
  - cffi=1.15.1=py39h74dc2b5_0
  - click=8.0.4=py39h06a4308_0
  - cryptography=38.0.1=py39h9ce1e76_0
  - dataclasses=0.8=pyh6d0b6a4_7
  - frozenlist=1.2.0=py39h7f8727e_0
  - giflib=5.2.1=h7b6447c_0
  - google-auth=2.6.0=pyhd3eb1b0_0
  - google-auth-oauthlib=0.4.4=pyhd3eb1b0_0
  - google-pasta=0.2.0=pyhd3eb1b0_0
  - grpc-cpp=1.46.1=h33aed49_0
  - grpcio=1.42.0=py39hce63b2e_0
  - h5py=3.7.0=py39h737f45e_0
  - hdf5=1.10.6=h3ffc7dd_1
  - icu=58.2=he6710b0_3
  - idna=3.4=py39h06a4308_0
  - importlib-metadata=4.11.3=py39h06a4308_0
  - jpeg=9e=h7f8727e_0
  - keras=2.9.0=py39h06a4308_0
  - keras-preprocessing=1.1.2=pyhd3eb1b0_0
  - krb5=1.19.2=hac12032_0
  - ld_impl_linux-64=2.38=h1181459_1
  - libblas=3.9.0=16_linux64_openblas
  - libcblas=3.9.0=16_linux64_openblas
  - libcurl=7.85.0=h91b91d3_0
  - libedit=3.1.20210910=h7f8727e_0
  - libev=4.33=h7f8727e_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=11.2.0=h1234567_1
  - libgfortran-ng=11.2.0=h00389a5_1
  - libgfortran5=11.2.0=h1234567_1
  - libgomp=11.2.0=h1234567_1
  - liblapack=3.9.0=16_linux64_openblas
  - libnghttp2=1.46.0=hce63b2e_0
  - libopenblas=0.3.21=h043d6bf_0
  - libpng=1.6.37=hbc83047_0
  - libprotobuf=3.20.1=h4ff587b_0
  - libssh2=1.10.0=h8f2d780_0
  - libstdcxx-ng=11.2.0=h1234567_1
  - markdown=3.3.4=py39h06a4308_0
  - multidict=6.0.2=py39h5eee18b_0
  - ncurses=6.3=h5eee18b_3
  - numpy-base=1.23.3=py39h1e6e340_0
  - oauthlib=3.2.1=py39h06a4308_0
  - openssl=1.1.1q=h7f8727e_0
  - opt_einsum=3.3.0=pyhd3eb1b0_1
  - packaging=21.3=pyhd3eb1b0_0
  - pip=22.2.2=py39h06a4308_0
  - pyasn1=0.4.8=pyhd3eb1b0_0
  - pyasn1-modules=0.2.8=py_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pyjwt=2.4.0=py39h06a4308_0
  - pyopenssl=22.0.0=pyhd3eb1b0_0
  - pyparsing=3.0.9=py39h06a4308_0
  - pysocks=1.7.1=py39h06a4308_0
  - python=3.9.13=haa1d7c7_2
  - python_abi=3.9=2_cp39
  - pyyaml=6.0=py39h7f8727e_1
  - re2=2022.04.01=h295c915_0
  - readline=8.2=h5eee18b_0
  - requests=2.28.1=py39h06a4308_0
  - requests-oauthlib=1.3.0=py_0
  - rsa=4.7.2=pyhd3eb1b0_1
  - setuptools=63.4.1=py39h06a4308_0
  - six=1.16.0=pyhd3eb1b0_1
  - snappy=1.1.9=h295c915_0
  - sqlite=3.39.3=h5082296_0
  - tensorboard=2.9.0=py39h06a4308_0
  - tensorboard-data-server=0.6.0=py39hca6d32c_0
  - tensorboard-plugin-wit=1.8.1=py39h06a4308_0
  - tensorflow=2.9.1=mkl_py39hb9fcb14_0
  - tensorflow-estimator=2.9.0=py39h06a4308_0
  - termcolor=1.1.0=py39h06a4308_1
  - tk=8.6.12=h1ccaba5_0
  - typing_extensions=4.3.0=py39h06a4308_0
  - tzdata=2022e=h04d1e81_0
  - urllib3=1.26.12=py39h06a4308_0
  - werkzeug=2.0.3=pyhd3eb1b0_0
  - wheel=0.37.1=pyhd3eb1b0_0
  - wrapt=1.14.1=py39h5eee18b_0
  - xz=5.2.6=h5eee18b_0
  - yaml=0.2.5=h7b6447c_0
  - yarl=1.8.1=py39h5eee18b_0
  - zipp=3.8.0=py39h06a4308_0
  - zlib=1.2.13=h5eee18b_0
  - pip:
    - blosc2==2.0.0
    - charset-normalizer==2.1.1
    - contourpy==1.0.5
    - cycler==0.11.0
    - cython==0.29.34
    - deepdish==0.3.5
    - flatbuffers==1.12
    - fonttools==4.38.0
    - gast==0.4.0
    - imageio==2.22.2
    - jinja2==3.1.2
    - joblib==1.2.0
    - jsonpatch==1.32
    - jsonpointer==2.3
    - kiwisolver==1.4.4
    - libclang==16.0.0
    - llvmlite==0.39.1
    - markupsafe==2.1.1
    - matplotlib==3.6.1
    - msgpack==1.0.5
    - multicoretsne==0.1
    - networkx==2.8.7
    - numba==0.56.4
    - numexpr==2.8.4
    - numpy==1.23.4
    - openai==0.27.0
    - opencv-python==4.6.0.66
    - pillow==9.2.0
    - protobuf==3.19.6
    - py-cpuinfo==9.0.0
    - python-dateutil==2.8.2
    - pywavelets==1.4.1
    - scikit-image==0.19.3
    - scikit-learn==1.1.3
    - scipy==1.9.3
    - tables==3.8.0
    - tcav==0.2.2
    - tensorflow-gpu==2.9.1
    - tensorflow-io-gcs-filesystem==0.32.0
    - threadpoolctl==3.1.0
    - tifffile==2022.10.10
    - torch==1.12.0+cu116
    - torch-cluster==1.6.0
    - torch-geometric==2.1.0.post1
    - torch-scatter==2.0.9
    - torch-sparse==0.6.15
    - torch-spline-conv==1.2.1
    - torchaudio==0.12.0+cu116
    - torchfile==0.1.0
    - torchnet==0.0.4
    - torchvision==0.13.0+cu116
    - tornado==6.2
    - tqdm==4.64.1
    - typing-extensions==4.4.0
    - visdom==0.2.3
    - websocket-client==1.4.2

## HardWare Requirement

- NVIDIA GTX3090 24G x1
- NVIDIA Driver Release 510.47.03
- CUDA_TOOL_KIT Release 11.6

<!-- ## Citation -->

<!-- If the dataset or code is of use to you, please consider citing: -->
