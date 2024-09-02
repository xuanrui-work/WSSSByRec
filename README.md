# WSSSByRec

This repository contains the implementation of the idea described in [Semantic Segmentation from Image Labels by Reconstruction from Structured Decomposition](./docs/Semantic_Segmentation_from_Image_Labels_by_Reconstruction_from_Structured_Decomposition.pdf), an approach to weakly supervised segmentation from image tags.


## Dataset

For our experiments, we use a custom dataset derived from the ImageNet-1K dataset. The dataset can be downloaded from [here](https://uofwaterloo-my.sharepoint.com/:u:/g/personal/x64zeng_uwaterloo_ca/EcfrK8hHh1JNrEL9tX80FMQBYbPJkKujw8n67pMd8Akf4A). It can also be generated from the original ImageNet-1K dataset using the notebook [preprocess_dataset.ipynb](./dataset/preprocess_dataset.ipynb).


## Getting started

### Prerequisites

Intall the required python packages using the below command:
```
pip3 install -r requirements.txt
```

Prepare the dataset following the below steps:
1. Download the dataset zip-file from the link provided above.
2. Extract all its contents into a directory named `dataset` at the root of the repository.

### Usage

Simply open the corresponding jupyter notebook with your favorite editor and run the cells. Training logs and visualizations will be displayed through tensorboard, which can be started by running the below command:
```
tensorboard --logdir=logs
```

1. To train the pretrained classifier $g$, run the notebook [Train Classifier.ipynb](./Train%20Classifier.ipynb).
2. To train the actual segmentation model $\{f_m, f_x\}$, run the notebook [Train Weak Segmentation.ipynb](./Train%20Weak%20Segmentation.ipynb).


## Results

### Results on training samples:

<img src="./docs/images/train-1-mask-overlay.png" width=500/>

### Results on validation samples:

<img src="./docs/images/val-1-mask-overlay.png" width=500/>



