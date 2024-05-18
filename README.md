# Requirements
CUDA >= 11.7

flash-atten >= 2.4.2

torch >= 1.13.1

apex

# [IJCAI 2024] Genomic-Selection

[![arXiv](https://img.shields.io/badge/arXiv-2405.09585-b31b1b.svg)](https://arxiv.org/abs/2405.09585) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This repository contains the code for our IJCAI 2024 in the AI and Social Good track paper `An Embarrassingly Simple Approach to Enhance Transformer Performance in Genomic Selection for Crop Breeding`. [[Paper](https://arxiv.org/abs/2405.09585)] 

![](fig/hikersgg.png)

## Environment

We test our codebase with PyTorch 1.12.0 with CUDA 11.6. Please install corresponding PyTorch and CUDA versions according to your computational resources.

Then install the rest of required packages by running `pip install -r requirements.txt`. This includes jupyter, as you need it to run the notebooks.

## Setup

We use the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset in this work, which consists of 108,077 images, each annotated with objects and relations. Following [previous work](https://arxiv.org/pdf/1701.02426.pdf), we filter the dataset to use the most frequent 150 object classes and 50 predicate classes for experiments.

You can download the images here, then extract the two zip files and put all the images in a single folder:

Part I: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

Part II: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

Then download VG metadata preprocessed by [IMP](https://arxiv.org/abs/1701.02426): [annotations](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5), [class info](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json),and [image metadata](http://svl.stanford.edu/projects/scene-graph/VG/image_data.json) and copy those three files in a single folder.

Finally, update `config.py` to with a path to the aforementioned data, as well as the absolute path to this directory.

We also provide two pre-trained weights:

1. The pre-trained Faster-RCNN checkpoint trained by [MotifNet](https://arxiv.org/pdf/1711.06640.pdf) from https://www.dropbox.com/s/cfyqhskypu7tp0q/vg-24.tar?dl=0 and place in `checkpoints/vgdet/vg-24`.

2. The pre-trained GB-Net checkpoint ``vgrel-11`` from https://github.com/alirezazareian/gbnet and place in `checkpoints/vgdet/vgrel-11`.

If you want to train from scratch, you can pre-train the model using Faster-RCNN checkpoint. However, we recommend to train from the GB-Net checkpoint.

## Usage

You can simply follow the instructions in the notebooks to run HiKER-SGG experiments:

1. For the PredCls task: ``train: ipynb/train_predcls/hikersgg_predcls_train.ipynb``, ``evaluate: ipynb/eval_predcls/hikersgg_predcls_test.ipynb``.
2. For the SGCls task: ``train: ipynb/train_sgcls/hikersgg_sgcls_train.ipynb``, ``evaluate: ipynb/eval_sgcls/hikersgg_sgcls_train.ipynb``.

Note that for the PredCls task, we start training from the GB-Net checkpoint; and for the SGCls task, we start training from the best PredCls checkpoint.

## Contact

If you have any questions, please  contact at [chenrenqi@pjlab.org.cn](mailto:chenrenqi@pjlab.org.cn).

## BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
Comming soon.
```
