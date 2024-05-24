# JIPNet
<img alt="PyTorch" height="25" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />
<a src="https://img.shields.io/badge/cs.CV-2405.03959-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2405.03959"> 
   <img src="https://img.shields.io/badge/cs.CV-2405.03959-b31b1b?logo=arxiv&logoColor=red"> 
</a> 

### 💬 This repo is the official implementation of:
- ***T-IFS 2024***: [Joint Identity Verification and Pose Alignment for Partial Fingerprints](https://arxiv.org/abs/2405.03959) 

[Xiongjun Guan](https://xiongjunguan.github.io/), Zhiyu Pan, Jianjiang Feng, Jie Zhou

<br>

## Introduction
Currently, portable electronic devices are becoming more and more popular. For lightweight considerations, their fingerprint recognition modules usually use limited-size sensors. However, partial fingerprints have few matchable features, especially when there are differences in finger pressing posture or image quality, which makes partial fingerprint verification challenging. Most existing methods regard fingerprint position rectification and identity verification as independent tasks, ignoring the coupling relationship between them -- relative pose estimation typically relies on paired features as anchors, and authentication accuracy tends to improve with more precise pose alignment. In this paper, we propose a novel framework for joint identity verification and pose alignment of partial fingerprint pairs, aiming to leverage their inherent correlation to improve each other. To achieve this, we present a multi-task CNN (Convolutional Neural Network)-Transformer hybrid network, and design a pre-training task to enhance the feature extraction capability. Experiments on multiple public datasets (NIST SD14, FVC2002 DB1A & DB3A, FVC2004 DB1A & DB2A, FVC2006 DB1A) and an in-house dataset show that our method achieves state-of-the-art performance in both partial fingerprint verification and relative pose estimation, while being more efficient than previous methods.

The overall flowchart of our proposed algorithm is shown as follows.
<br>
<p align="center">
    <img src="./images/flowchart.png"/ width=90%> <br />
</p>
<br>

The structure of **JIPNet** (the name `JIP` stands for **J**oint **I**dentity Verification and **P**ose Alignment for Partial Fingerprints) is shown as follows.
<br>
<p align="center">
    <img src="./images/method.png"/ width=90%> <br />
</p>
<br>


## News
Code will be released after this paper is officially accepted.

<br>
  

## Citation
If you find this repository useful, please give us stars and use the following BibTeX entry for citation.
```
@article{guan2024joint,
  author={Guan, Xiongjun and Pan, Zhiyu and Feng, Jianjiang and Zhou, Jie},
  journal={arXiv preprint arXiv:2405.03959},
  title={Joint Identity Verification and Pose Alignment for Partial Fingerprints}, 
  year={2024},
```


## License
This project is released under the MIT license. Please see the LICENSE file for more information.

## Contact me

If you have any questions about the code, please contact Xiongjun Guan gxj21@mails.tsinghua.edu.cn
