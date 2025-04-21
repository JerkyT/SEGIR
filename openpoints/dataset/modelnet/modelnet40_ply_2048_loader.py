"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
from examples.classification.aug import *

import random

def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path

def load_data(data_dir, partition, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        with h5py.File(h5_name, 'r') as f: # ['data', 'faceId', 'label', 'normal']
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            # faceId = f['faceId'][:]#.astype('int64')
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    all_index = np.array([i for i in range(len(all_data))])
    return all_data, all_label, all_index

def FPS(point, npoints = 1024):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoints,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, centroids.astype(np.int32)

@DATASETS.register_module()
class ModelNet40Ply2048(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'modelnet40_ply_hdf5_2048'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/ModelNet40Ply2048",
                 split='train',
                 transform=None,
                 transformc=None,
                 shapley = False,
                 cfg = None
                 ):

        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir

        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.num_points = num_points
        self.transform = transform
        self.transformc = transformc if self.partition == 'train' else None
        self.shapley = shapley
        self.cfg = cfg
        data_index = cfg.data_index
        self.cutoff_frequency = cfg.cutoff_frequency
        self.gft = cfg.gft
        self.m = cfg.m
        self.geometry_mode = cfg.geometry_mode

        self.data, self.label, self.index = load_data(data_dir, self.partition, self.url)
        self.data = self.data
        if data_index:
            idx = np.load(data_index)
            self.data, self.label = self.data[idx], self.label[idx]

        if self.shapley:
            if self.partition == 'train':
                self.shapleys = np.load('./data/shaply_ModelNet40Ply2048/PointNet2Encoder_ST/train1000/final_p.npy').mean(-1) # (N, 1024, C)
            else:
                self.shapleys = np.load('./data/shaply_ModelNet40Ply2048/PointNet2Encoder_ST/test1000/final_p.npy').mean(-1) # (N, 1024, C)

        # print(self.partition)
        # print(self.transform)
        # print(self.transformc.transforms)
        # exit()

        if self.gft:
            self.v_reals = np.load('/home/liweigang/PointMetaBase/data/GFT/ModelNet40_real_'+ str(self.partition) + '.npy') # (N, n, n)
            self.fv = np.load('/home/liweigang/PointMetaBase/data/GFT/Feature_vector_'+ str(self.partition) + '.npy') # (N, n, n)

        if is_class_in_list('Graph_Domain_Geometry_Enhancement', self.transform) or is_class_in_list('Graph_Domain_Geometry_Enhancement', self.transformc):
            self.v_reals = np.load('/home/liweigang/PointMetaBase/data/GFT/ModelNet40_real_'+ str(self.partition) + '.npy') # (N, n, n)
            self.fv = np.load('/home/liweigang/PointMetaBase/data/GFT/Feature_vector_'+ str(self.partition) + '.npy') # (N, n, n)

            self.sk = np.load('/home/liweigang/PointMetaBase/data/ModelNet40Ply2048/ModelNet40_2048_' + self.geometry_mode +  '_' + str(self.partition) + ".npy")
            self.v_reals_sk = np.load('/home/liweigang/PointMetaBase/data/GFT/'+ self.geometry_mode + '_real_'+ str(self.partition) + '.npy')
            self.fv_sk = np.load('/home/liweigang/PointMetaBase/data/GFT/Feature_vector_' + self.geometry_mode + '_'+ str(self.partition) + '.npy') # (N, n, n)

            if self.m == 'shapely':
                self.shapleys = np.load('./data/shaply_ModelNet40Ply2048/PointNet2Encoder_ST/train1000/final_fft_p.npy').mean(-1) # (N, 1024, C)
            if data_index:
                self.sk = self.sk[idx]
                self.v_reals = self.v_reals[idx]
                self.fv = self.fv[idx]
                self.v_reals_sk = self.v_reals_sk[idx]
                self.fv_sk = self.fv_sk[idx]

        if is_class_in_list('Spatial_geometry_Enhancement', self.transform) or is_class_in_list('Spatial_geometry_Enhancement', self.transformc):
            if not hasattr(self, 'sk'):
                self.sk = np.load('/home/liweigang/PointMetaBase/data/ModelNet40Ply2048/ModelNet40_2048_' + self.geometry_mode +  '_' + str(self.partition) + ".npy")

        self.mask = 0.1
        logging.info(f'==> sucessfully loaded {self.partition} data')

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points] # N * 3
        label = self.label[item]

        if self.partition == 'train':
            index = np.arange(pointcloud.shape[0])
            np.random.shuffle(index)
            pointcloud = pointcloud[index]

        data = {
            'pos': pointcloud.copy(),
            'y': label,
            'index': item
        }

        if self.shapley and self.partition == 'train':
            data['shapely'] = self.shapleys[item][index].copy()
        
        if self.gft:
            v_real, fv = torch.from_numpy(self.v_reals[item:item+1]), torch.from_numpy(self.fv[item:item+1])
            data['v_real'], data['fv'] = v_real, fv

        if is_class_in_list('Graph_Domain_Geometry_Enhancement', self.transform) or is_class_in_list('Graph_Domain_Geometry_Enhancement', self.transformc):
            data['sk'] = self.sk[item]
            v_real, fv = torch.from_numpy(self.v_reals[item:item+1]), torch.from_numpy(self.fv[item:item+1])
            data['v_real'], data['fv'] = v_real, fv
            v_real_sk, fv_sk = torch.from_numpy(self.v_reals_sk[item:item+1]), torch.from_numpy(self.fv_sk[item:item+1])
            data['v_real_sk'], data['fv_sk'] = v_real_sk, fv_sk

        if is_class_in_list('Spatial_geometry_Enhancement', self.transform) or is_class_in_list('Spatial_geometry_Enhancement', self.transformc):
            data['sk'] = self.sk[item]

        if self.partition == 'train' and self.transformc is not None:
            data = self.transformc(data)
            if 'heights' in data.keys():
                data['xc'] = torch.cat((data['pos'], data['heights']), dim=1)
            else:
                data['xc'] = data['pos']

        if self.transform is not None:
            data['pos'] = pointcloud.copy()
            data = self.transform(data)

        ####
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']

        # 需要保留的键列表
        keys_to_keep = ['x', 'pos', 'y', 'shapely', 'xc', 'posc', 'v_real', 'fv', 'index']

        # 过滤字典，保留指定的键
        filtered_data  = {key: value for key, value in data.items() if key in keys_to_keep}

        return filtered_data 

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return 40

# if __name__ == '__main__':

#     def knn(points, k):
#         b, n, _ = points.shape
#         points_transpose = points.transpose(2, 1)
#         inner = -2 * torch.matmul(points, points_transpose)
#         xx = torch.sum(points ** 2, dim=-1, keepdim=True)
#         pairwise_distance = -xx - inner - xx.transpose(2, 1)
#         _, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)
#         return idx

#     def eig_vector(data, K):
#         b, n, _ = data.shape
#         idx = knn(data, k=K)  # idx (b,n,K)

#         idx0 = torch.arange(0, b, device=data.device).reshape((b, 1)).expand(-1, n*K).reshape((1, b*n*K))
#         idx1 = torch.arange(0, n, device=data.device).reshape((1, n, 1)).expand(b, n, K).reshape((1, b*n*K))
#         idx = idx.reshape((1, b*n*K))
#         idx = torch.cat([idx0, idx1, idx], dim=0)  # (3, b*n*K)
        
#         ones = torch.ones(idx.shape[1], dtype=torch.float32, device=data.device)
#         A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense()  # (b,n,n)
        
#         # Modify this line to ensure A is symmetrical using a float-friendly method
#         A = torch.maximum(A, A.transpose(1, 2))
        
#         deg = torch.diag_embed(torch.sum(A, dim=2))
#         laplacian = deg - A
#         u, v = torch.linalg.eigh(laplacian)  # (b,n,n)
#         return v, laplacian, u # .real

#     def GFT(pc_ori, K = 8, v_real = None, x_ = None):
#         x = pc_ori.unsqueeze(0)
#         # x = torch.from_numpy(pc_ori[np.newaxis,:])  # Assuming this is already in the shape (b,n,3)
#         b, n, _ = x.shape

#         if v_real is None:
#             v_real, _, _ = eig_vector(x, K)  # Extract v.real from the tuple
#         if x_ is None:
#             x_ = torch.einsum('bij,bjk->bik', v_real.transpose(1,2), x)  # Use v_real here

#         x = torch.einsum('bij,bjk->bik', v_real, x_) # Use v_real again here

#         return x.numpy(), x_.numpy(), v_real.numpy()

#     def normalize_array(arr):
#         # 矩阵归一化
#         min_val = np.min(arr)
#         max_val = np.max(arr)
#         normalized_arr = (arr - min_val) / (max_val - min_val)
#         return 2 * normalized_arr - 1

#     from scipy.special import sph_harm

#     dir_name = 'modelnet40_ply_hdf5_2048'
#     md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
#     url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
#     data, label, index = load_data("./data/ModelNet40Ply2048", 'train', url)
#     l_points, feathers, v_reals = [], [], []
#     for points in data:
        
#         points = points[0:1024, :]
#         ##### sht
#         # 将 3D 点转换为球坐标系
#         x = points[:, 0]
#         y = points[:, 1]
#         z = points[:, 2]
#         r = np.sqrt(x**2 + y**2 + z**2)
#         theta = np.arccos(z / r)  # 极角
#         phi = np.arctan2(y, x)  # 方位角

#         # 计算球谐函数 (这里我们使用 n=4 和 m=2 的球谐函数作为示例)
#         n = 40
#         m = 20
#         Ynm = sph_harm(m, n, phi, theta)

#         # 将球谐函数的结果应用到原始点云上
#         # 这里我们通过将球谐函数结果与原点云坐标的乘积作为示例
#         transformed_points = np.zeros_like(points)
#         transformed_points[:, 0] = r * np.abs(Ynm) * x
#         transformed_points[:, 1] = r * np.abs(Ynm) * y
#         transformed_points[:, 2] = r * np.abs(Ynm) * z

#         transformed_points = normalize_array(transformed_points)

#         # 
#         x, x_, v_real = GFT(torch.from_numpy(transformed_points))
#         # print(v_real.shape)
#         feathers.append(x_)
#         v_reals.append(v_real)
#         l_points.append(transformed_points[np.newaxis,:])
    
#     ##### sht
#     sht_save = np.concatenate(l_points, axis=0)
#     feathers_save = np.concatenate(feathers, axis=0)
#     v_real = np.concatenate(v_reals, axis=0)
#     np.save('./sht_train.npy', sht_save)
#     np.save('/home/liweigang/PointMetaBase/data/GFT/Feature_vector_SHT_train.npy', feathers_save)
#     np.save('/home/liweigang/PointMetaBase/data/GFT/SHT_real_train.npy', v_real)