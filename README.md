# [IJCAI 2024] Genomic-Selection

[![arXiv](https://img.shields.io/badge/arXiv-2405.09585-b31b1b.svg)](https://arxiv.org/abs/2405.09585) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This repository contains the code for our IJCAI 2024 in the AI and Social Good track paper `An Embarrassingly Simple Approach to Enhance Transformer Performance in Genomic Selection for Crop Breeding`. [[Paper](https://arxiv.org/abs/2405.09585)] 

![](fig/hikersgg.png)

## Environment

We test our codebase with PyTorch 1.13.1 with CUDA 11.7. It is better to install a higher version for flash attention.

Then install:
``flash-atten >= 2.4.2``,
``apex``

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
@misc{chen2024embarrassingly,
      title={An Embarrassingly Simple Approach to Enhance Transformer Performance in Genomic Selection for Crop Breeding}, 
      author={Renqi Chen and Wenwei Han and Haohao Zhang and Haoyang Su and Zhefan Wang and Xiaolei Liu and Hao Jiang and Wanli Ouyang and Nanqing Dong},
      year={2024},
      eprint={2405.09585},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
