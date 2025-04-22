# Robust 3D Models through Skeleton-Enhanced Geometric Information Reinforcement

## State
This work is under submission to ["IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT"](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=19).

## Prerequisites
Install necessary packages using:
```bash
bash install.sh
```

Install PyGeM
```bash
cd PyGen
python setup.py install
cd ..
```

## Data
Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), [ModelNet40-C](https://github.com/jiachens/ModelNet40-C), and [PointCloud-C](https://drive.google.com/uc?id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm) datasets and put them in the Data directory:
```plaintext
SEGIR/
└── data/
    ├── ModelNet40Ply2048
        └── modelnet40_ply_hdf5_2048
    ├── modelnet40_c
    └── pointcloud_c
```
## Generate or download the ‘Signals in the point cloud domain’ corresponding to modelNet
You can use this project to generate the point cloud skeletons for the ModelNet40 dataset by yourself: [Point2Skeleton](https://github.com/clinplayer/Point2Skeleton)

Or you can also download it directly from this [link](https://drive.google.com/drive/folders/1vzz-3QjIQ8VBdwMa32cQGAi8aLmPp1RH?usp=sharing).
You need to put the file in the specified directory：
```plaintext
SEGIR/
└── data/
    ├── ModelNet40Ply2048
        ├── modelnet40_ply_hdf5_2048
        └── ModelNet40_2048_SK_train.npy
```
## Generate or download the Point Cloud Skeletons corresponding to modelNet
Run the following command to generate the point cloud and skeleton signals in the atlas domain:
```plaintext
python ./examples/classification/Gen_GFT.py
```
Or you can also download it directly from this [link](https://drive.google.com/drive/folders/1vzz-3QjIQ8VBdwMa32cQGAi8aLmPp1RH?usp=sharing).
You need to put the file in the specified directory：
```plaintext
SEGIR/
└── data/
    └── GFT
        ├── Feature_vector_train.npy
        ├── ModelNet40_real_train.npy
        ├── Feature_vector_SK_train.npy
        └── SK_real_train.npy
```
## Script
Train PointNet++ model using ST method:
```bash
bash ./script/trainST.sh
```
Train PointNet++ model using DesenAT method:
```bash
bash ./script/train.sh
```
Testing in ModelNet40-C 
```bash
bash ./script/test_modelnetC.sh
```
Testing in PointCloud-C
```bash
bash ./script/test_pointcloudC.sh
```

## Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints) and [PointNeXt](https://github.com/guochengqian/PointNeXt). We integrated [APES](https://github.com/JunweiZheng93/APES) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase) into the code. We also have integrated methods for handling corrupted point clouds into our code, thanks to the excellent work of [ModelNet-C](https://github.com/jiachens/ModelNet40-C) ,[PointCloud-C](https://github.com/ldkong1205/PointCloud-C), [GSDA](https://github.com/WoodwindHu/GSDA) and [Point2Skeleton](https://github.com/clinplayer/Point2Skeleton).
