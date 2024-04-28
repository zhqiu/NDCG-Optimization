# NDCG-Optimization

Code to reproduce the results in paper "[Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence](https://arxiv.org/abs/2202.12183)". The proposed methods in this paper, SONG and K-SONG, are implemented in [LibAUC](https://libauc.org/).

## Updates

Building on **SONG** and **K-SONG**, we add novel algorithms with *faster convergence rates*: **Faster SONG<sup>v1/v2</sup>/K-SONG<sup>v1/v2</sup>**. Additionally, we incorporatd Precision@K and top-K mAP optimization algorithms, validated through experiments on molecular data (code available under the ``Mol-exps`` directory).

## Citation  
@inproceedings{ICML:2022:NDCG,  
author = {Zi-Hao Qiu and Quanqi Hu and Yongjian Zhong and Lijun Zhang and Tianbao Yang},  
title = {Large-scale Stochastic Optimization of {NDCG} Surrogates for Deep Learning with Provable Convergence},  
booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML)},  
pages = {18122--18152},  
year = {2022},  
}
