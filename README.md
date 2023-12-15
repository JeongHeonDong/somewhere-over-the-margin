# Somewhere Over the Margin

In metric learning field, margin-based hinging is the golden rule. In this project, we will try to overcome the margin-based hinging by using a new loss function.

## Dependencies

- python 3.8.16
- pytorch 2.1.0
- torchvision 0.16.0
- pytorch-metric-learning 2.3.0
- record-keeper 0.9.32
- tensorboard 2.12.1
- matplotlib 3.2.2
- numpy 1.24.4
- umap-learn 0.5.5
- faiss 1.7.2

you can find the full list of dependencies in `dependencies.txt`.

### Setup Guide

```base
$ conda create -n cv-project python=3.8.16
$ conda activate cv-project
$ conda install pytorch==2.1.0 torchvision==0.16.0 cudatoolkit=<Your Cuda Version> -c pytorch -y
$ conda install -c conda-forge pytorch-metric-learning -y
$ conda install matplotlib tensorboard -y
$ conda install -c conda-forge umap-learn -y
$ conda install -c pytorch faiss-cpu=1.7.2 mkl=2021 blas=1.0=mkl -y
$ pip install record-keeper
```

## Reference

- pytorch-metric-learning: [🔗 Link](https://github.com/KevinMusgrave/pytorch-metric-learning)
