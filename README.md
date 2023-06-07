# OKD-CL

This is the repository for the OKD-CL dataset, associated with the NeurIPS 2023 Datasets and Benchmarks submission, <em>``Object Knowledge Decomposition and Components Label Dataset for Knowledge Guided Object Recognition''</em>.

## Download the dataset

You can download the dataset directly from the realese/OKD-CL.tar.gz in this page. 

The dataset should contain directories for train,test,hard_masks,instance-level_masks and component-level masks. (The knowledge vectors had been restored in code source as 99 .txt files of the knowledge directory). Train and test directory both have 99 sub-directories including 99 object categories' image instances and directories of masks also have the same sub-directories formed including corresponding pixel-level annotation. 

Note that we assume the researcher has access to Hard-ImageNet; the files above only contain the data of OKD-CL (image instances,3 kinds of masks and knowledge vectors in the code source) we collected. You can download Hard-ImageNet follow this [link to download from box
(https://umd.box.com/s/gx5qx4w03dgsumjclo7wpbdqov4xxrly). 

## Setting up the data

1. Donwloads the dataset files and unzip them locally.
2. Downloads the checkpoint files and unzip them locally. 
3. Modified the dataset dirs in dataset/utiles.py.

## Evaluate Models

We provide training codes for all 7 models, corresponding parameter settings and corresponding model weights can be downloaded in realese/SaveWeights.tar.gz include (VGG16, IncptionV1, MobileNetV1, MobileNetV2, ResNet50, ResNet101, ResNet152). We provide codes for all 4 evaluation methods including (ACC & FRR metrics evaluation, masking & noise perturbation test, method migration evaluation, input transformation and activation offset evaluation).

1. ACC & FRR metrics evaluation is in the train_modelname.py intergrate as a function named Metrics_Test with the model training code.
2. masking & noise perturbation test is in the disturbance.py(*related parameters defined in utiles/para.py).
3. method migration evaluation is in a independent sub-directory named Hard-ImageNet Knowledge Guidance in the same level directory with other evaluation code(This sub-directory has a similar file structure to the parent directory).
4. input transformation and activation offset evaluation is in the transformation_test.py.

<!-- ## Citation -->

<!-- If the dataset or code is of use to you, please consider citing: -->

