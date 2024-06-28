# [IJCAI 2024] Genomic-Selection

[![arXiv](https://img.shields.io/badge/arXiv-2405.09585-b31b1b.svg)](https://arxiv.org/abs/2405.09585) 

## Introduction

This repository contains the code for our IJCAI 2024 in the AI and Social Good track paper `An Embarrassingly Simple Approach to Enhance Transformer Performance in Genomic Selection for Crop Breeding`. [[Paper](https://arxiv.org/abs/2405.09585)] 

We release the employed dataset rice3k at . Please feel free to use it. 

However, we are very sorry to inform you that the Wheat dataset is limited by the need for our partners to use it in another article, so we currently do not have the permission to open source it. But we will open source in the future.

## Environment

We run our code with PyTorch 1.13.1 with CUDA 11.7. It is better to install a higher version for flash attention.

Then install:
``flash-atten >= 2.4.2``,
``apex``

## Usage

You can simply follow the instruction to train and evaluate:

``bash distributed_train.sh``.

Note that our model is a simple end-to-end training.

## Contact

If you have any questions, please  contact at [chenrenqi@pjlab.org.cn](mailto:chenrenqi@pjlab.org.cn).

## BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
@article{chen2024embarrassingly,
  title={An Embarrassingly Simple Approach to Enhance Transformer Performance in Genomic Selection for Crop Breeding},
  author={Chen, Renqi and Han, Wenwei and Zhang, Haohao and Su, Haoyang and Wang, Zhefan and Liu, Xiaolei and Jiang, Hao and Ouyang, Wanli and Dong, Nanqing},
  journal={arXiv preprint arXiv:2405.09585},
  year={2024}
}
```
