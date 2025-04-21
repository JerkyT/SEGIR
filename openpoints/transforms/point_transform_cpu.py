import numpy as np
import torch
from .point_transformer_gpu import DataTransforms
from scipy.linalg import expm, norm
from scipy.special import sph_harm

@DataTransforms.register_module()
class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if key == 'index':
                continue
            if not torch.is_tensor(data[key]):
                if str(data[key].dtype) == 'float64':
                    data[key] = data[key].astype(np.float32)
                data[key] = torch.from_numpy(np.array(data[key]))
        return data

@DataTransforms.register_module()
class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1], **kwargs):
        self.angle = angle

    def __call__(self, data):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        data['pos'] = np.dot(data['pos'], np.transpose(R))
        return data


@DataTransforms.register_module()
class RandomRotateZ(object):
    def __init__(self, angle=1.0, rotate_dim=2, random_rotate=True, **kwargs):
        self.angle = angle * np.pi
        self.random_rotate = random_rotate
        axis = np.zeros(3)
        axis[rotate_dim] = 1
        self.axis = axis
        self.rotate_dim = rotate_dim

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if self.random_rotate:
            rotate_angle = np.random.uniform(-self.angle, self.angle)
        else:
            rotate_angle = self.angle
        R = self.M(self.axis, rotate_angle)
        data['pos'] = np.dot(data['pos'], R)  # anti clockwise
        return data

    def __repr__(self):
        return 'RandomRotate(rotate_angle: {}, along_z: {})'.format(self.rotate_angle, self.along_z)


@DataTransforms.register_module()
class RandomScale(object):
    def __init__(self, scale=[0.8, 1.2],
                 scale_anisotropic=False,
                 scale_xyz=[True, True, True],
                 mirror=[-1, -1, -1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.anisotropic = scale_anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        if len(scale) == 1:
            scale = scale.repeat(3)
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        data['pos'] *= scale
        return data

    def __repr__(self):
        return 'RandomScale(scale_low: {}, scale_high: {})'.format(self.scale_min, self.scale_max)


@DataTransforms.register_module()
class RandomScaleAndJitter(object):
    def __init__(self,
                 scale=[0.8, 1.2],
                 scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 scale_anisotropic=False,  # scaling in different ratios for x, y, z
                 jitter_sigma=0.01, jitter_clip=0.05,
                 mirror=[-1, -1, -1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale = scale
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.scale_xyz = scale_xyz
        self.noise_sigma = jitter_sigma
        self.noise_clip = jitter_clip
        self.anisotropic = scale_anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)

        if len(scale) == 1:
            scale = scale.repeat(3)
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        jitter = np.clip(self.noise_sigma * np.random.randn(data['pos'].shape[0], 3), -self.noise_clip, self.noise_clip)
        data['pos'] = data['pos'] * scale + jitter
        return data


@DataTransforms.register_module()
class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
        self.shift = shift

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data['pos'] += shift
        return data

    def __repr__(self):
        return 'RandomShift(shift_range: {})'.format(self.shift_range)


@DataTransforms.register_module()
class RandomScaleAndTranslate(object):
    def __init__(self,
                 scale=[0.9, 1.1],
                 shift=[0.2, 0.2, 0],
                 scale_xyz=[1, 1, 1],
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.shift = shift

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        scale *= self.scale_xyz

        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data['pos'] = np.add(np.multiply(data['pos'], scale), shift)

        return data


@DataTransforms.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['pos'][:, 0] = -data['pos'][:, 0]
        if np.random.rand() < self.p:
            data['pos'][:, 1] = -data['pos'][:, 1]
        return data


@DataTransforms.register_module()
class RandomJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.noise_sigma = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        jitter = np.clip(self.noise_sigma * np.random.randn(data['pos'].shape[0], 3), -self.noise_clip, self.noise_clip)
        data['pos'] += jitter
        return data


@DataTransforms.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None, **kwargs):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.rand() < self.p:
            lo = np.min(data['x'][:, :3], 0, keepdims=True)
            hi = np.max(data['x'][:, :3], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data['x'][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data['x'][:, :3] = (1 - blend_factor) * data['x'][:, :3] + blend_factor * contrast_feat
            """vis
            from openpoints.dataset import vis_points
            vis_points(data['pos'], data['x']/255.)
            """
        return data


@DataTransforms.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05, **kwargs):
        self.p = p
        self.ratio = ratio

    def __call__(self, data):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data['x'][:, :3] = np.clip(tr + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005, **kwargs):
        self.p = p
        self.std = std

    def __call__(self, data):
        if np.random.rand() < self.p:
            noise = np.random.randn(data['x'].shape[0], 3)
            noise *= self.std * 255
            data['x'][:, :3] = np.clip(noise + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]

        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, **kwargs):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(data['x'][:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        data['x'][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data


@DataTransforms.register_module()
class RandomDropFeature(object):
    def __init__(self, feature_drop=0.2,
                 drop_dim=[0, 3],
                 **kwargs):
        self.p = feature_drop
        self.dim = drop_dim

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['x'][:, self.dim[0]:self.dim[-1]] = 0
        return data


@DataTransforms.register_module()
class NumpyChromaticNormalize(object):
    def __init__(self,
                 color_mean=None,
                 color_std=None,
                 **kwargs):

        self.color_mean = np.array(color_mean).astype(np.float32) if color_mean is not None else None
        self.color_std = np.array(color_std).astype(np.float32) if color_std is not None else None

    def __call__(self, data):
        if data['x'][:, :3].max() > 1:
            data['x'][:, :3] /= 255.
        if self.color_mean is not None:
            data['x'][:, :3] = (data['x'][:, :3] - self.color_mean) / self.color_std
        return data

@DataTransforms.register_module()
class Supervoxel(object):
    def __init__(self, **kwargs):
        self.npoints = 1024
        pass

    def __call__(self, data):
        weights = data['shapely']
        weights -= min(weights)
        idx = np.argsort(weights)[::-1]

        p = 0
        s = np.sum(weights)
        for i in range(len(idx) - 1, -1, -1):
            p += weights[idx[i]]
            if p > 0.005 * s:
                idx = idx[i::]
                break

        # 将距离参考点小于0.2的点标记为True
        distances = np.linalg.norm(data['pos'] - data['pos'][idx][:, np.newaxis], axis=2)
        radius = np.mean(np.sort(distances, 1)[:, 1:10])
        mask = np.any(distances <= radius, axis=0) # 距离小的点

        data['pos'] = data['pos'][~mask]

        npoints = self.npoints
        point  = data['pos']
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
        data['pos'] = point

        return data

#################################################################
##  Generate Various Common Corruptions ###
import numpy as np
from . import distortion
from .util import *

# np.random.seed(2021)
ORIG_NUM = 1024
import torch
import random
### Transformation ###
'''
Rotate the point cloud
'''

def rotation(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    gamma = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    beta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.

    matrix_1, matrix_2, matrix_3 = np.zeros((B,3,3)),np.zeros((B,3,3)),np.zeros((B,3,3))
    matrix_1[:,0,0], matrix_1[:,1,1], matrix_1[:,1,2], matrix_1[:,2,1], matrix_1[:,2,2], = \
                                                            1, np.cos(theta), -np.sin(theta), np.sin(theta),np.cos(theta)
    matrix_2[:,0,0], matrix_2[:,0,2], matrix_2[:,1,1], matrix_2[:,2,0], matrix_2[:,2,2], = \
                                                            np.cos(gamma), np.sin(gamma), 1, -np.sin(gamma), np.cos(gamma)
    matrix_3[:,0,0], matrix_3[:,0,1], matrix_3[:,1,0], matrix_3[:,1,1], matrix_3[:,2,2], = \
                                                            np.cos(beta), -np.sin(beta), np.sin(beta), np.cos(beta), 1

    # matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    # matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
    # matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')

    return new_pc # normalize(new_pc)

'''
Shear the point cloud
'''
def shear(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]

    b = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    d = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    e = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    f = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)

    matrix = np.zeros((B, 3, 3))
    matrix[:,0,0], matrix[:,0,1], matrix[:,0,2] = 1, 0, b
    matrix[:,1,0], matrix[:,1,1], matrix[:,1,2] = d, 1, e
    matrix[:,2,0], matrix[:,2,1], matrix[:,2,2] = f, 0, 1

    new_pc = np.matmul(pointcloud, matrix).astype('float32')
    return new_pc #normalize(new_pc)

'''
Scale the point cloud
'''

def scale(pointcloud, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    pointcloud = torch.from_numpy(pointcloud.transpose(0,2,1))
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            B x C x N array, original batch of point clouds
        Return:
            B x C x N array, scaled batch of point clouds
    """
    scales = (torch.rand(pointcloud.shape[0], 3, 1) * 2. - 1.) * c + 1.
    pointcloud *= scales
    pointcloud = pointcloud.numpy().transpose(0,2,1)
    return pointcloud
### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud,severity = 1):
    #TODO
    B, N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c,c,(B, N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc # normalize(new_pc)

'''
Add Gaussian noise to point cloud
'''
def gaussian_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(B, N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    # new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add noise to the edge-length-2 cude
'''
def background_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    c = N // 10
    jitter = np.random.uniform(-1,1,(B, c, C))
    pointcloud[:,N - c::,:] = jitter
    # new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    return pointcloud #normalize(new_pc)

'''
Upsampling
'''
def upsampling(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//5, N//4, N//3, N//2, N][severity-1]
    c = N // 10
    index = np.random.choice(ORIG_NUM, (B, c), replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05,0.05, (B, c, C))
    pointcloud[:,N - c::,:] = add
    # new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
    return pointcloud # normalize(new_pc)
    
'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    c = N // 60    
    for i in range(B):
        index = np.random.choice(ORIG_NUM, c, replace=False)
        pointcloud[i,index] += np.random.choice([-1,1], size=(c,C)) * 0.1

    return pointcloud #normalize(pointcloud)

### Point Number Modification ###

'''
Uniformly sampling the point cloud
'''
def uniform_sampling(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    # c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][severity-1]
    c = N // 30
    index = np.random.choice(ORIG_NUM, (B, ORIG_NUM - c), replace=False)
    return pointcloud[index]

def ffd_distortion(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
    for i in range(B):
        pointcloud[i] = normalize(distortion.distortion(pointcloud[i].copy(), severity=c))
    return pointcloud

def rbf_distortion(pointcloud, severity = 1):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')

def rbf_distortion_inv(pointcloud, severity = 1):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='inv_multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')

def height(batch_data, severity = 1):
    batch_data = batch_data
    height = np.min(batch_data, axis = 1)
    batch_data = batch_data - height[:,np.newaxis,:]

    return batch_data

def original(pointcloud, severity = 1):
    return pointcloud

'''
Cutout several part in the point cloud
'''
def cutout(pointcloud,severity = 1):
    B, N, C = pointcloud.shape
    c = [(2,30), (3,30), (5,30), (7,30), (10,30)][severity-1]
    for j in range(B):
        for _ in range(c[0]):
            i = np.random.choice(pointcloud[j].shape[0],1)
            picked = pointcloud[j][i]
            dist = np.sum((pointcloud[j] - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            pointcloud[j][idx.squeeze()] = 0
        # pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud

'''
Density-based up-sampling the point cloud
'''
def density_inc(pointcloud, severity):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    # idx = np.random.choice(N,c[0])
    # 
    for j in range(B):
        temp = []
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            # idx = idx[idx_2]
            temp.append(p_temp[idx.squeeze()])
            p_temp = np.delete(p_temp, idx.squeeze(), axis=0)

        idx = np.random.choice(p_temp.shape[0],1024 - c[0] * c[1])
        temp.append(p_temp[idx.squeeze()])
        p_temp = np.concatenate(temp)

        pointcloud[j] = p_temp

    return pointcloud

'''
Density-based sampling the point cloud
'''
def density(pointcloud, severity = 1):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]

    for j in range(B):
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            idx = idx[idx_2]
            # p_temp = np.delete(p_temp, idx.squeeze(), axis=0)
            p_temp[idx.squeeze()] = 0

        pointcloud[j] = p_temp
    return pointcloud

def shapely(pointcloud, severity = 1):
    B, N, C = pointcloud.shape

    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(B, N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    # new_pc = np.clip(new_pc,-1,1)
    return new_pc

def get_trans(use_trans = True):
    MAP = {
        "Density":{
            # 'cutout': cutout,
            'density': density,
            'density_inc': density_inc,
            'original' : original, # 不变
            # 'occlusion': occlusion,
            # 'lidar': lidar,
        },
        "noise":{
            'uniform': uniform_noise,
            'gaussian': gaussian_noise,
            'impulse': impulse_noise,
            'original': original,
            # 'upsampling': upsampling,
            # 'background': background_noise,
            # 'original' : original, # 不变
        },
        "Transformation":{
            # 'rotation': rotation,
            'original': original,
            'shear': shear,
        },
        'size' : {
            'scale': scale,
            'original': original,
        },
        # 'height' : height,
    }

    trans_list = [original]

    if use_trans:
        trans_Transformation = random.choice(list(MAP['Transformation']))
        trans_noise = random.choice(list(MAP['noise']))
        trans_Density = random.choice(list(MAP['Density']))
        # trans_size = random.choice(list(MAP['size']))

        # trans_list = [MAP["Transformation"][trans_Transformation], MAP["noise"][trans_noise], MAP["Density"][trans_Density], MAP["size"][trans_size]]
        trans_list = [MAP["Transformation"][trans_Transformation], MAP["noise"][trans_noise], MAP["Density"][trans_Density]] + [original] # MAP["size"][trans_size]
    return trans_list

@DataTransforms.register_module()
class TND(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        data['pos'] = data['pos'][np.newaxis]
        trans_list = get_trans(True)
        for f in trans_list:
            data['pos'][:,:,0:3] = f(data['pos'][:,:,0:3], severity = np.random.randint(1, 6))

        data['pos'] = data['pos'][0]

        return data

@DataTransforms.register_module()
class FarthestPS(object): # farthest_point_sample
    def __init__(self, **kwargs):
        self.npoints = 1024

    def __call__(self, data):
        if data['pos'].shape[0] == self.npoints:
            pass
        else:
            npoints = self.npoints
            point  = data['pos']
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
            data['pos'] = point
        data['pos2'] = data['pos'].copy()

        return data

def trans(points, NT_trans_list, trans_trans_list):
    """
    points B N 3
    """
    points = points.numpy()

    if len(trans_trans_list) > 0:
        points_trans = points.copy()
        for f in trans_trans_list:
            points_trans = f(points_trans, severity = np.random.randint(1, 6))

        points_trans = torch.from_numpy(points_trans.transpose(0,2,1)).cuda()

    else:
        points_trans = None

    for f in NT_trans_list:
        points = f(points)
    points = torch.from_numpy(points.transpose(0,2,1)).cuda()

    return points, points_trans

def normalize_point_cloud_batch(pc):
    """
    将点云的几何中心归一化到原点（支持批处理）
    :param pc: 输入的点云数据，形状为 (B, N, 3)，
               B为批次大小，N为每个点云的点数量
    :return: 归一化后的点云，形状为 (B, N, 3)
    """
    # 计算每个批次点云的几何中心 (B, 1, 3)
    center = np.mean(pc, axis=1, keepdims=True)
    
    # 将点云平移，使几何中心重合到原点 (B, N, 3)
    normalized_pc = pc - center
    
    return normalized_pc, center

@DataTransforms.register_module()
class Spatial_geometry_Enhancement(object):
    def __init__(self, **kwargs):
        self.n_neighbors = 1
        pass

    def __call__(self, data):
        point_cloud, skeleton_points = data['pos'], data['sk']

        # point_cloud = point_cloud.numpy()
        # skeleton_points = skeleton_points.numpy()
        
        #
        point_cloud, center_p = normalize_point_cloud_batch(point_cloud)
        skeleton_points, center_sk = normalize_point_cloud_batch(skeleton_points)
        #

        # 计算每一个骨架点到原始点云中每一个点的距离
        distances = np.linalg.norm(skeleton_points[:, np.newaxis, :] - point_cloud[np.newaxis, :, :], axis=2)
        # 对每一个骨架点，找到最近的前n个原始点
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        # print(nearest_indices.shape)
        # 创建一个布尔数组来记录哪些原始点被标记
        marked_points = np.zeros(point_cloud.shape[0], dtype=bool)

        # 标记找到的最近点
        for i in range(nearest_indices.shape[0]):
            marked_points[nearest_indices[i]] = True

        # 获取未被标记的原始点的索引
        marked_indices = np.where(marked_points)
        unmarked_indices = np.where(~marked_points)

        # print(marked_indices)
        adv_point = np.concatenate([point_cloud[unmarked_indices], point_cloud[marked_indices]], axis = 0) # 去除远点
        # adv_point = np.concatenate([point_cloud[marked_indices], point_cloud[unmarked_indices]], axis = 0) # 去除近点

        data['pos'] = adv_point # + center_p
        return data

@DataTransforms.register_module()
class PointWOLF(object):
    def __init__(self, **kwargs):
        self.num_anchor = 4 #args.w_num_anchor
        self.sample_type = 'fps' # args.w_sample_type
        self.sigma = 0.5 # args.w_sigma

        self.R_range = (-abs(10), abs(10))
        self.S_range = (1., 3)
        self.T_range = (-abs(0.25), abs(0.25))

    def __call__(self, data):
        """
        input :
            pos([N,3])
            
        output : 
            pos([N,3]) : original pointcloud
            pos_new([N,3]) : Pointcloud augmneted by PointWOLF
        """
        pos = data['pos']
        M=self.num_anchor #(Mx3)
        N, _=pos.shape #(N)
        
        if self.sample_type == 'random':
            idx = np.random.choice(N,M)#(M)
        elif self.sample_type == 'fps':
            idx = self.fps(pos, M) #(M)
        
        pos_anchor = pos[idx] #(M,3), anchor point
        
        pos_repeat = np.expand_dims(pos,0).repeat(M, axis=0)#(M,N,3)
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype)  #(M,N,3)
        
        #Move to canonical space
        pos_normalize = pos_repeat - pos_anchor.reshape(M,-1,3)
        
        #Local transformation at anchor point
        pos_transformed = self.local_transformaton(pos_normalize) #(M,N,3)
        
        #Move to origin space
        pos_transformed = pos_transformed + pos_anchor.reshape(M,-1,3) #(M,N,3)
        
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)
        pos_new = self.normalize(pos_new)
        
        data['pos'] = pos_new.astype('float32')
        return data
        

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N,3])
            pos_anchor([M,3])
            pos_transformed([M,N,3])
            
        output : 
            pos_new([N,3]) : Pointcloud after weighted local transformation 
        """
        M, N, _ = pos_transformed.shape
        
        #Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor,1).repeat(N, axis=1) - np.expand_dims(pos,0).repeat(M, axis=0) #(M,N,3), d
        
        project_axis = self.get_random_axis(1)

        projection = np.expand_dims(project_axis, axis=1)*np.eye(3)#(1,3,3)
        
        #Project distance
        sub = sub @ projection # (M,N,3)
        sub = np.sqrt(((sub) ** 2).sum(2)) #(M,N)  
        
        #Kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))  #(M,N) 
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0) #(N,3)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T) # normalize by weight
        return pos_new

    
    def fps(self, pos, npoint):
        """
        input : 
            pos([N,3])
            npoint(int)
            
        output : 
            centroids([npoints]) : index list for fps
        """
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_) #(M)
        distance = np.ones(N, dtype=np.float64) * 1e10 #(N)
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    
    def local_transformaton(self, pos_normalize):
        """
        input :
            pos([N,3]) 
            pos_normalize([M,N,3])
            
        output :
            pos_normalize([M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        M,N,_ = pos_normalize.shape
        transformation_dropout = np.random.binomial(1, 0.5, (M,3)) #(M,3)
        transformation_axis =self.get_random_axis(M) #(M,3)

        degree = np.pi * np.random.uniform(*self.R_range, size=(M,3)) / 180.0 * transformation_dropout[:,0:1] #(M,3), sampling from (-R_range, R_range) 
        
        scale = np.random.uniform(*self.S_range, size=(M,3)) * transformation_dropout[:,1:2] #(M,3), sampling from (1, S_range)
        scale = scale*transformation_axis
        scale = scale + 1*(scale==0) #Scaling factor must be larger than 1
        
        trl = np.random.uniform(*self.T_range, size=(M,3)) * transformation_dropout[:,2:3] #(M,3), sampling from (1, T_range)
        trl = trl*transformation_axis
        
        #Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3) # scailing factor to diagonal matrix (M,3) -> (M,3,3)
        #Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)
        
        pos_normalize = pos_normalize@R@S + trl.reshape(M,1,3)
        return pos_normalize
    
    def get_random_axis(self, n_axis):
        """
        input :
            n_axis(int)
            
        output :
            axis([n_axis,3]) : projection axis   
        """
        axis = np.random.randint(1,8, (n_axis)) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz    
        m = 3 
        axis = (((axis[:,None] & (1 << np.arange(m)))) > 0).astype(int)
        return axis
    
    def normalize(self, pos):
        """
        input :
            pos([N,3])
        
        output :
            pos([N,3]) : normalized Pointcloud
        """
        pos = pos - pos.mean(axis=-2, keepdims=True)
        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        pos = scale * pos
        return pos

def inverse_spherical_harmonics_transform(coefficients, theta, phi):
    """
    通过球谐系数恢复球面函数值

    :param coefficients: 球谐系数点云数据，N x 3 的数组，包含 n, m 及其对应的系数 a_{nm}
    :param theta: 极角（球坐标系中的第一个角度），数组形状为 (N,)
    :param phi: 方位角（球坐标系中的第二个角度），数组形状为 (N,)
    :return: 恢复后的球面函数 f(theta, phi)
    """
    f = np.zeros_like(theta)  # 初始化恢复函数

    # 遍历球谐系数点云
    for coef in coefficients:
        n = int(coef[0])  # 阶数 n
        m = int(coef[1])  # 次数 m
        a_nm = coef[2]    # 球谐系数 a_{nm}
        
        # 计算球谐基函数 Y_{nm}(\theta, \phi)
        Ynm = sph_harm(m, n, phi, theta)  # 计算对应的球谐函数值
        
        # 将系数加权与球谐基函数的值
        f += a_nm * Ynm.real  # 使用实部进行累加
    
    return f

@DataTransforms.register_module()
class SHT(object):
    def __init__(self, **kwargs):
        pass
    
    def normalize_array(self, arr):
        # 矩阵归一化
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return 2 * normalized_arr - 1

    def __call__(self, data):
        points = data['pos']

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # 极角
        phi = np.arctan2(y, x)  # 方位角

        n = 4
        m = 2
        Ynm = sph_harm(m, n, phi, theta)

        transformed_points = np.zeros_like(points)
        transformed_points[:, 0] = r * np.abs(Ynm) * x
        transformed_points[:, 1] = r * np.abs(Ynm) * y
        transformed_points[:, 2] = r * np.abs(Ynm) * z

        transformed_points = self.normalize_array(transformed_points)
        data['sht'] = transformed_points
        return data

@DataTransforms.register_module()
class Spatial_geometry_Enhancement_SHT(object):
    def __init__(self, **kwargs):
        self.n_neighbors = 1
        pass

    def __call__(self, data):
        point_cloud, skeleton_points = data['pos'], data['sht']
        # 计算每一个骨架点到原始点云中每一个点的距离
        distances = np.linalg.norm(skeleton_points[:, np.newaxis, :] - point_cloud[np.newaxis, :, :], axis=2)
        # 对每一个骨架点，找到最近的前n个原始点
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        # print(nearest_indices.shape)
        # 创建一个布尔数组来记录哪些原始点被标记
        marked_points = np.zeros(point_cloud.shape[0], dtype=bool)

        # 标记找到的最近点
        for i in range(nearest_indices.shape[0]):
            marked_points[nearest_indices[i]] = True

        # 获取未被标记的原始点的索引
        marked_indices = np.where(marked_points)
        unmarked_indices = np.where(~marked_points)

        # print(marked_indices)
        adv_point = np.concatenate([point_cloud[unmarked_indices], point_cloud[marked_indices]], axis = 0)

        data['pos'] = adv_point
        return data