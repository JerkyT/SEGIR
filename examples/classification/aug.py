import numpy as np
import torch
import sys
sys.path.append("./emd/")
import emd_module as emd
# from torch.utils.tensorboard import SummaryWriter
import random

def normalize_array_torch(arr):
    # 矩阵归一化
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def chamfer_distance(pc1, pc2):
    """
    Calculate the Chamfer Distance between two point clouds.
    Input:
        pc1: Point cloud 1, [B, N, C]
        pc2: Point cloud 2, [B, M, C]
    Output:
        dist: Chamfer Distance, scalar
    """
    # Compute pairwise distances
    dist1 = square_distance_torch(pc1, pc2)  # [B, N, M]
    dist2 = square_distance_torch(pc2, pc1)  # [B, M, N]

    # Find the minimum distance from each point in pc1 to pc2
    dist1_min = torch.min(dist1, dim=2)[0]  # [B, N]
    dist2_min = torch.min(dist2, dim=2)[0]  # [B, M]

    print(dist1_min.shape, dist2_min.shape)
    # Sum the minimum distances
    chamfer_dist = torch.mean(dist1_min, dim = 1) + torch.mean(dist2_min, dim = 1)

    return chamfer_dist

def square_distance_torch(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(2)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)

    return dist

def cutmix_r(data_batch, BETA = 1., PROB = 0.5, num_points = 1024):
    r = np.random.rand(1)
    if BETA > 0 and r < PROB:
        lam = np.random.beta(BETA, BETA)
        B = data_batch['x'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['y']
        target_b = data_batch['y'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['x']
        point_b = data_batch['x'][rand_index]
        point_c = data_batch['x'][rand_index]
        # point_a, point_b, point_c = point_a.to(device), point_b.to(device), point_c.to(device)

        remd = emd.emdModule()
        remd = remd.cuda()

        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(1024 * lam)
        int_lam = max(1, int_lam)
        gamma = np.random.choice(num_points, int_lam, replace=False, p=None)
        for i2 in range(B):
            data_batch['x'][i2, gamma, :] = point_c[i2, gamma, :]

        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / num_points
        # points = data_batch['pc'].transpose(2, 1)
        data_batch['y2'] = target_b
        data_batch['lam'] = lam

    return data_batch
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam

def cutmix_k(data_batch, BETA = 1., PROB = 0.5, num_points = 1024):
    r = np.random.rand(1)
    if BETA > 0 and r < PROB:
        lam = np.random.beta(BETA, BETA)
        B = data_batch['x'].size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = data_batch['y']
        target_b = data_batch['y'][rand_index]

        point_a = torch.zeros(B, 1024, 3)
        point_b = torch.zeros(B, 1024, 3)
        point_c = torch.zeros(B, 1024, 3)
        point_a = data_batch['x']
        point_b = data_batch['x'][rand_index]
        point_c = data_batch['x'][rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

        int_lam = int(num_points * lam)
        int_lam = max(1, int_lam)

        random_point = torch.from_numpy(np.random.choice(1024, B, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(B))
        query = point_a[ind1, random_point].view(B, 1, 3)
        dist = torch.sqrt(torch.sum((point_a - query.repeat(1, num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(B):
            data_batch['x'][i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / num_points
        # points = points.transpose(2, 1)
        # pred, trans_feat = model(points)
        # loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam
        data_batch['y2'] = target_b
        data_batch['lam'] = lam
        
    return data_batch

def mixup(data_batch, MIXUPRATE = 0.4):

    batch_size = data_batch['x'].size()[0]
    idx_minor = torch.randperm(batch_size)
    mixrates = (0.5 - np.abs(np.random.beta(MIXUPRATE, MIXUPRATE, batch_size) - 0.5))
    label_main = data_batch['y']
    label_minor = data_batch['y'][idx_minor]
    label_new = torch.zeros(batch_size, 40)
    for i in range(batch_size):
        if label_main[i] == label_minor[i]: # same label
            label_new[i][label_main[i]] = 1.0
        else:
            label_new[i][label_main[i]] = 1 - mixrates[i]
            label_new[i][label_minor[i]] = mixrates[i]
    label = label_new

    data_minor = data_batch['x'][idx_minor]
    mix_rate = torch.tensor(mixrates).float()
    mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

    mix_rate_expand_xyz = mix_rate.expand(data_batch['x'].shape).cuda()

    remd = emd.emdModule()
    remd = remd.cuda()
    _, ass = remd(data_batch['x'], data_minor, 0.005, 300)
    ass = ass.long()
    for i in range(batch_size):
        data_minor[i] = data_minor[i][ass[i]]
    data_batch['x'] = data_batch['x'] * (1 - mix_rate_expand_xyz) + data_minor * mix_rate_expand_xyz
    data_batch['y2'] = label_minor
    data_batch['lam'] = torch.tensor(mix_rate).squeeze_().cuda()

    return data_batch

def rsmix(data, BETA = 1., PROB = 0.5, n_sample=512, KNN=False):
    cut_rad = np.random.beta(BETA, BETA)
    data_batch = data['x'].cpu().numpy()
    label_batch = data['y'].cpu().numpy()

    rand_index = np.random.choice(data_batch.shape[0],data_batch.shape[0], replace=False) # label dim : (16,) for model
    
    if len(label_batch.shape) == 1:
        label_batch = np.expand_dims(label_batch, axis=1)
        
    label_a = label_batch[:,0]
    label_b = label_batch[rand_index][:,0]
        
    data_batch_rand = data_batch[rand_index] # BxNx3(with normal=6)
    rand_idx_1 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    rand_idx_2 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    if KNN:
        knn_para = min(int(np.ceil(cut_rad*n_sample)),n_sample)
        pts_erase_idx, query_point_1 = cut_points_knn(data_batch, rand_idx_1, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points_knn(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_2 x 3(or 6)
    else:
        pts_erase_idx, query_point_1 = cut_points(data_batch, rand_idx_1, cut_rad, nsample=n_sample) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample) # B x num_points_in_radius_2 x 3(or 6)
    
    query_dist = query_point_1[:,:,:3] - query_point_2[:,:,:3]
    
    pts_replaced = np.zeros((1,data_batch.shape[1],data_batch.shape[2]))
    lam = np.zeros(data_batch.shape[0],dtype=float)

    for i in range(data_batch.shape[0]):
        if pts_erase_idx[i][0][0]==data_batch.shape[1]:
            tmp_pts_replaced = np.expand_dims(data_batch[i], axis=0)
            lam_tmp = 0
        elif pts_add_idx[i][0][0]==data_batch.shape[1]:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            dup_points_idx = np.random.randint(0,len(tmp_pts_erased), size=len(pts_erase_idx_tmp))
            tmp_pts_replaced = np.expand_dims(np.concatenate((tmp_pts_erased, data_batch[i][dup_points_idx]), axis=0), axis=0)
            lam_tmp = 0
        else:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_tmp = np.unique(pts_add_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_ctrled_tmp = pts_num_ctrl(pts_erase_idx_tmp,pts_add_idx_tmp)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            # input("INPUT : ")
            tmp_pts_to_add = np.take(data_batch_rand[i], pts_add_idx_ctrled_tmp, axis=0)
            tmp_pts_to_add[:,:3] = query_dist[i]+tmp_pts_to_add[:,:3]
            
            tmp_pts_replaced = np.expand_dims(np.vstack((tmp_pts_erased,tmp_pts_to_add)), axis=0)
            
            lam_tmp = len(pts_add_idx_ctrled_tmp)/(len(pts_add_idx_ctrled_tmp)+len(tmp_pts_erased))
        
        pts_replaced = np.concatenate((pts_replaced, tmp_pts_replaced),axis=0)
        lam[i] = lam_tmp
    
    data_batch_mixed = np.delete(pts_replaced, [0], axis=0)    
        
    data['x'] = torch.FloatTensor(data_batch_mixed).cuda()
    data['y'] = torch.tensor(label_a).cuda()
    data['y2'] = torch.tensor(label_b).cuda()
    data['lam'] = torch.tensor(lam).cuda()

    return data

def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape # S=1
    
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    sqrdists = square_distance(query, xyz) # Bx1,N #제곱거리
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B,1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i]>knn_dist[i][0]]=N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def cut_points_knn(data_batch, idx, radius, nsample=512, k=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = knn_points(k=k, xyz=data_batch[:,:,:3], query=query_points[:,:,:3], nsample=nsample)
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def cut_points(data_batch, idx, radius, nsample=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = query_ball_point_for_rsmix(radius, nsample, data_batch[:,:,:3], query_points[:,:,:3])
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    # device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)
    
    return dist

def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    '''
        input : pts - to erase 
                pts - to add
        output :pts - to add (number controled)
    '''
    if len(pts_erase_idx)>=len(pts_add_idx):
        num_diff = len(pts_erase_idx)-len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(pts_add_idx, pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)])
    else:
        pts_add_idx_ctrled = np.sort(np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False))
    return pts_add_idx_ctrled

def get_m(cfg):
    if 'ST_' in cfg.pretrained_path:
        m = '_ST'
    elif 'AT_' in cfg.pretrained_path:
        m = '_AT'
    elif 'TN_' in cfg.pretrained_path:
        m = '_TN'
    elif 'cutmix_r' in cfg.pretrained_path:
        m = '_cutmix_r'
    elif 'cutmix_k' in cfg.pretrained_path:
        m = '_cutmix_k'
    elif 'mixup' in cfg.pretrained_path:
        m = '_mixup'
    elif 'rsmix' in cfg.pretrained_path:
        m = '_rsmix'
    elif 'round_' in cfg.pretrained_path:
        m = '_round'
    elif 'Quantify_' in cfg.pretrained_path:
        m = '_Quantify'
    elif 'Supervoxel_' in cfg.pretrained_path:
        m = '_Supervoxel'
    elif 'TND_' in cfg.pretrained_path:
        m = '_TND'
    elif 'TQ_' in cfg.pretrained_path:
        m = '_TQ'
    elif '_ASE' in cfg.pretrained_path:
        m = '_ASE'
    elif 'FFT' in cfg.pretrained_path:
        m = '_FFT'
    elif 'Decoupling' in cfg.pretrained_path:
        m = '_Decoupling'
    else:
        m = '_M1'
    return m

def vis_weights(model):
    ########
    # 可视化
    writer = SummaryWriter('./AT')
    for i, (name, param) in enumerate(model.named_parameters()):
        # print(name, '   ', param.shape)

        weights = param.clone().cpu().data.numpy()
        if len(weights.shape) >= 2:
            for j in range(weights.shape[1]): 
                writer.add_histogram(name , weights[:,j], j)
        # if name == 'module.sa1.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa1.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa2.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa2.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa2.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa2.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa3.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa3.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa3.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa3.mlp_convs.1.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa4.mlp_convs.0.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa4.mlp_convs.0.weight' , weights[:,j,:,:], j)
        # if name == 'module.sa4.mlp_convs.1.weight':
        #     weights = param.clone().cpu().data.numpy()
        #     for j in range(weights.shape[1]): 
        #         writer.add_histogram('module.sa4.mlp_convs.1.weight' , weights[:,j,:,:], j)
    writer.close()
    ########

def Energy(fft):
    return np.abs(fft**2).mean(-1)

def knn(points, k):
    b, n, _ = points.shape
    points_transpose = points.transpose(2, 1)
    inner = -2 * torch.matmul(points, points_transpose)
    xx = torch.sum(points ** 2, dim=-1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)
    return idx

@torch.no_grad()
def eig_vector(data, K):
    b, n, _ = data.shape
    idx = knn(data, k=K)  # idx (b,n,K)

    idx0 = torch.arange(0, b, device=data.device).reshape((b, 1)).expand(-1, n*K).reshape((1, b*n*K))
    idx1 = torch.arange(0, n, device=data.device).reshape((1, n, 1)).expand(b, n, K).reshape((1, b*n*K))
    idx = idx.reshape((1, b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0)  # (3, b*n*K)
    
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense()  # (b,n,n)
    
    # Modify this line to ensure A is symmetrical using a float-friendly method
    A = torch.maximum(A, A.transpose(1, 2))
    
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A
    u, v = torch.linalg.eigh(laplacian)  # (b,n,n)
    return v, laplacian, u # .real

def random_mask(n = 1024, cutoff_frequency = 0.75):
    cutoff_frequency = 1 - cutoff_frequency
    mask = np.ones(n)
    idx = np.arange(0, n)
    np.random.shuffle(idx)
    idx = idx[0:int(cutoff_frequency * n)]
    mask1 = np.random.rand(len(idx)) + 0.5
    mask[idx] = mask1
    return mask

# def shapely_mask(shapelys, cutoff_frequency = 1):
#     cutoff_frequency = 1 - cutoff_frequency
#     mask = np.ones(len(shapelys))
#     idx = np.argsort(shapelys)[::-1][0:int(cutoff_frequency * len(shapelys))]
#     mask1 = np.random.rand(len(idx)) + 0.5
#     mask[idx] = mask1

#     return mask

def shapely_mask(shapelys, cutoff_frequency = 1):
    mask = np.ones(len(shapelys))
    if cutoff_frequency == 1:
        return mask
    cutoff_frequency = 1 - cutoff_frequency
    # idx = np.argsort(shapelys)[::-1][0:int(cutoff_frequency * len(shapelys))]
    
    n = len(shapelys)
    idx = np.arange(0, n)
    np.random.shuffle(idx)
    idx = idx[0:int(cutoff_frequency * n)]
    shapelys = shapelys[idx]

    mask1 = (shapelys - shapelys.min()) / ((shapelys - shapelys.min()).max() + 0.0000001) + 0.5
    mask[idx] = mask1

    return mask

# 反傅里叶变化
def iGFT(pc_ori, x_, K = 8, v_real = None):
    x_ = x_.unsqueeze(0)
    x = torch.from_numpy(pc_ori[np.newaxis,:]) # Assuming this is already in the shape (b,n,3)

    if v_real is not None:
        v_real, _, _ = eig_vector(x, K)  # Extract v.real from the tuple

    x = torch.einsum('bij,bjk->bik', v_real, x_)  # Use v_real again here

    return x[0]

def GFT(pc_ori, cutoff_frequency = 0.0, K = 8, m = None, shapely = None, v_real = None, x_ = None):
    # cutoff_frequency 保留的比例
    x = torch.from_numpy(pc_ori[np.newaxis,:])  # Assuming this is already in the shape (b,n,3)
    # x = pc_ori
    b, n, _ = x.shape

    if v_real is None:
        v_real, _, _ = eig_vector(x, K)  # Extract v.real from the tuple
    if x_ is None:
        x_ = torch.einsum('bij,bjk->bik', v_real.transpose(1,2), x)  # Use v_real here

    x_or = x_.clone()
    if m is not None:
        # mask = np.ones_like(pc_ori[:, 0])
        # # 攻击方式
        # if m == 'h':
        #     mask[0:int((1 - cutoff_frequency) * n)] = 0
        # # lowpass
        # elif m == 'l':
        #     mask[int(cutoff_frequency * n)::] = 0
        # # random
        # elif m == 'r':
        #     # mask0, mask1, mask2, mask3 = random_mask(10, 1), random_mask(90, cutoff_frequency), random_mask(800, 1), random_mask(124, 1)
        #     # mask = np.concatenate([mask0, mask1, mask2, mask3])
        #     mask0, mask1 = random_mask(10, 1), random_mask(1014, 0)
        #     mask = np.concatenate([mask0, mask1])
        #     mask = mask.astype(np.float32)
        # # shapely
        # if m == 'shapely':
        #     idx = 10
        #     idx2 = 90
        #     low, mid = shapely[0:idx], shapely[idx::] #, shapely[idx2::]

        #     mask1, mask2 = shapely_mask(low, 1), shapely_mask(mid, 0) # , shapely_mask(hight, 0)
        #     mask = np.concatenate([mask1, mask2])
        #     mask = mask.astype(np.float32)

        # # Energy
        # if m == 'energy_R':
        #     # Total energy in the sequence
        #     energy = Energy(x_[0])
        #     total_energy = torch.sum(energy)

        #     # Set a threshold for selection
        #     threshold = cutoff_frequency * total_energy

        #     indices = list(range(0, len(energy)))
        #     selected_energy = 0.

        #     for idx in indices:
        #         if selected_energy <= threshold:
        #             selected_energy += energy[idx]
        #         else:
        #             break

        #     mask0, mask1 = random_mask(idx, 1), random_mask(1024 - idx, 0)
        #     mask = np.concatenate([mask0, mask1])
        #     mask = mask.astype(np.float32)

        # x_ = x_ * mask[np.newaxis,:,np.newaxis]
        x = torch.einsum('bij,bjk->bik', v_real, x_) # Use v_real again here

    # return x[0], v_real

    return x[0], x_[0], x_or, v_real

# 生成均匀分布的新点
def generate_points_along_lines(vertices, lines, target_points_count):
    current_points_count = len(vertices)
    new_points_needed = target_points_count - current_points_count
    
    if new_points_needed <= 0:
        return vertices  # 已经有足够多的点了，不需要生成新点
    
    line_lengths = []
    for line in lines:
        start_vertex = vertices[line[0]]
        end_vertex = vertices[line[1]]
        length = np.linalg.norm(start_vertex - end_vertex)
        line_lengths.append(length)
    
    total_length = sum(line_lengths)
    points_per_length_unit = new_points_needed / total_length
    
    new_points = []
    for line, length in zip(lines, line_lengths):
        start_vertex = vertices[line[0]]
        end_vertex = vertices[line[1]]
        num_new_points = int(length * points_per_length_unit)
        
        for i in range(1, num_new_points + 1):
            t = i / (num_new_points + 1)
            new_point = (1 - t) * start_vertex + t * end_vertex
            new_points.append(new_point)
    
    # If we still need more points, distribute them evenly across all lines
    while len(new_points) + current_points_count < target_points_count:
        for line, length in zip(lines, line_lengths):
            if len(new_points) + current_points_count >= target_points_count:
                break
            start_vertex = vertices[line[0]]
            end_vertex = vertices[line[1]]
            t = np.random.rand()
            new_point = (1 - t) * start_vertex + t * end_vertex
            new_points.append(new_point)
    
    return np.vstack((vertices, new_points[:new_points_needed]))

def normalize_batch(arr_batch):
    # 矩阵归一化
    min_val = torch.min(arr_batch, dim=1, keepdim=True)[0]
    max_val = torch.max(arr_batch, dim=1, keepdim=True)[0]
    normalized_batch = (arr_batch - min_val) / (max_val - min_val)
    return normalized_batch

def normalize_point_cloud(pc):
    # 中心化处理
    pc[:, 0] -= (torch.max(pc[:, 0]) + torch.min(pc[:, 0])) / 2
    pc[:, 1] -= (torch.max(pc[:, 1]) + torch.min(pc[:, 1])) / 2
    pc[:, 2] -= (torch.max(pc[:, 2]) + torch.min(pc[:, 2])) / 2
    
    # 计算各维度的长度
    leng_x = torch.max(pc[:, 0]) - torch.min(pc[:, 0])
    leng_y = torch.max(pc[:, 1]) - torch.min(pc[:, 1])
    leng_z = torch.max(pc[:, 2]) - torch.min(pc[:, 2])
    
    # 计算缩放比例
    max_leng = max(leng_x, leng_y, leng_z)
    ratio = 2.0 / max_leng

    # 归一化
    pc *= ratio

    return pc

def remove_sparse_points(pc, threshold_ratio=0.1):
    # 计算每个点到原点的距离
    distances = torch.sqrt(torch.sum(pc**2, dim=1))
    
    # 找到需要删除的点的距离阈值
    threshold_distance = torch.quantile(distances, 1 - threshold_ratio)

    outlier_indices = np.where(distances > threshold_distance)

    pc[outlier_indices] = pc[outlier_indices] * (threshold_distance / distances[outlier_indices]).reshape(-1, 1)
        
    return pc

# 图谱域中，点云骨架的攻击
def Graph_Domain_Skeleton_Attack_Dataload(data):
    a = random.uniform(0.0, 0.3) #random.uniform(0.0, 0.3)
    point_cloud, point_sk, v_real, v_real_sk, fv, fv_sk = data['pos'], data['sk'], data['v_real'], data['v_real_sk'], data['fv'], data['fv_sk']
    x, x_, x_or = GFT(point_cloud, v_real = v_real, x_ = fv)
    xsk, xsk_, xsk_or = GFT(point_sk, v_real = v_real_sk, x_ = fv_sk)
    # xsk_or[100::] = normalize_array_torch(xsk_or[100::]) + 0.5
    # new_s = x_or
    new_s = (1 - a) * x_or + a * xsk_or
    # new_s[100::] = new_s[100::] * a * xsk_or[100::]
    adv_point = iGFT(point_cloud, new_s)
    # adv_point = normalize_point_cloud(adv_point)
    # adv_point = remove_sparse_points(adv_point)
    # adv_point = normalize_point_cloud(adv_point)

    return adv_point

def is_class_in_list(class_name, transform_obj):
    """
    判断class_name是否在class_list中
    """
    if transform_obj is None:
        return False
    try:
        class_list = []
        for t in transform_obj.transforms:
            class_list.append(get_full_class_name(t).split('.')[-1])

        if class_name in class_list:
            return True
        else:
            return False
    except:
        return False

def get_full_class_name(obj):
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"