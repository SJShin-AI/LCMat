# LCMat (Loss Curvature Matching for Dataset Selection and Condensation, AISTATS 2023)

Official PyTorch implementation of
**["Loss-Curvature Matching for Dataset Selection and Condensation"](https://arxiv.org/pdf/2303.04449v1.pdf)** (AISTATS 2023) by
[Seungjae Shin*](https://sites.google.com/view/seungjae-shin),
[HeeSun Bae*](https://sites.google.com/view/baeheesun),
[Donghyeok Shin](https://aailab.kaist.ac.kr/xe2/members_phdstudent/20976),
[Weonyoung Joo](https://scholar.google.co.kr/citations?user=r2eJgW4AAAAJ&hl=ko&oi=ao),
and [Il-Chul Moon](https://aailab.kaist.ac.kr/xe2/members_professor/6749).

## Overview

LCMat identifies the optimal dataset by matching the worst loss-curvature gap between the original dataset and the reduced dataset.
It learns toward achieving the generalization around the local parameter region on dataset reduction procedure. Our implementation code is largely dependent on the code of [DeepCore](https://github.com/PatrickZH/DeepCore). We thank the authors for providing these codes.

<p align="center">
  <img 
    width="700"
    src="https://user-images.githubusercontent.com/105624747/219567990-beb0cbb7-0ebb-44bd-957f-7182a79af8ab.png"
  >
</p>

Here, $\theta$ is a parameter, $\rho$ denotes the maximum size of the perturbation. LCMat matches the Loss Curvature of the training dataset, $T$, and the resulting dataset, $S$, by reducing the gap between the curvature of $\mathcal{l}(T)$ and that of $\mathcal{l}(S)$.

By considering the sharpness on loss difference, LCMat(right) can successfully identify the reduced dataset $S$ matching the loss landscape of original $T$, although the subset selected by Craig(left) does not match the loss curvature of $T$.

<p align="center">
  <img 
    width="700"
    src="https://user-images.githubusercontent.com/105624747/219572052-fded6505-861d-4db6-bf4e-f9cde1c5a71b.png"
  >
</p>

## Setup
Please install required libraries as follows.

We kindly suggest other researchers to run this code on `python = 3.8` version.
```
pip install -r requirements.txt
```

## Reproduce
For reproduce the results of LCMAT-S, we provide a bash file for running `main.py`, which located at: 
```
/bash/LCMat_XXX.sh
```
Here, XXX is dataset. You can get results in `result/` directory.

You can also reproduce cross-architecture generalization result by running `cross_network_generalization.py`.

We will also release the code of LCMat-C soon.

Thank you for your Interest in our paper!
