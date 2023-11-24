# Somewhere Over the Margin

In metric learning field, margin-based hinging is the golden rule. In this project, we will try to overcome the margin-based hinging by using a new loss function.

## Dependencies

- python 3.8.16
- pytorch 1.6.0
- torchvision 0.7.0
- pytorch-metric-learning 2.3.0
- record-keeper 0.9.32
- tensorboard 2.10.0
- matplotlib 3.5.3
- numpy 1.23.5
- umap-learn 0.5.4
- faiss 1.7.4

you can find the full list of dependencies in `dependencies.txt`.

### Setup Guide

```base
$ conda create -n cv-project python=3.8.16
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge pytorch-metric-learning
$ conda install matplotlib tensorboard
$ conda install -c conda-forge umap-learn
$ conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
$ pip install record-keeper
```

## Reference

- pytorch-metric-learning: [🔗 Link](https://github.com/KevinMusgrave/pytorch-metric-learning)
