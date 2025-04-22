"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets.utils import extract_archive, check_integrity
from aug import *

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

data, label, index = load_data(data_dir = "./data/ModelNet40Ply2048", partition = 'train', url = f'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')

# np.save('/home/liweigang/PointMetaBase/data/GFT/ModelNet40_2048.npy', data)
# np.save('/home/liweigang/PointMetaBase/data/GFT/ModelNet40_Label.npy', label)

data = np.load('./data/ModelNet40Ply2048/ModelNet40_2048_SK_train.npy')[:, 0, :, :]

# Feature vector
v_list = []
real_list = []
for point in data:
    point = point[0:1024, :]
    _, _, x_, v_real = GFT(point)
    v_list.append(x_)
    real_list.append(v_real)

np.save('./data/GFT/Feature_vector_train.npy', np.concatenate(v_list, 0))
np.save('./data/GFT/ModelNet40_real_train.npy', np.concatenate(real_list, 0))
