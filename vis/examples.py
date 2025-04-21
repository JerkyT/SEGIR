import os
import numpy as np
import pickle
from config import data_dir, def_ranges
import matplotlib.pyplot as plt
# 设置 Matplotlib 使用 WenQuanYi Micro Hei 字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from PIL import Image
from dataload import ModelNet40_C
from load_clean import ModelNet40Ply2048
example_ids = [6,7,32,54,85]
severity = 4
examples_file = "examples.pkl"

c_list = ["rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv", "uniform", "gaussian", "impulse", "upsampling", "background", "occlusion", "lidar", "density", "density_inc", "cutout"]

def normalize(new_pc):
    new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
    new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
    new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc

def rotation(pointcloud,severity = 5):
    N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
    gamma = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
    beta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.

    matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
    
    new_pc = np.matmul(pointcloud,matrix_1)
    new_pc = np.matmul(new_pc,matrix_2)
    new_pc = np.matmul(new_pc,matrix_3).astype('float32')

    return normalize(new_pc)

def build_examples(example_ids, severity = 4):
    all_examples = {}
    for example_id in example_ids:
        examples = []
        for corruption in list(def_ranges["corruption"].keys()):
            if corruption == "none":
                continue
            file_path = os.path.join(data_dir, "data_{}_{:d}.npy".format(corruption, severity))
            examples.append(np.load(file_path)[example_id,:])
        all_examples[example_id] = examples

    with open(examples_file, "wb") as f:
        pickle.dump(all_examples, f)
    return examples

def load_examples():
    with open(examples_file, 'rb') as f:
        all_examples = pickle.load(f)
    return all_examples

def rotation_matrix(pitch, yaw, roll):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R

def draw_one_example(example, rotate=[0, 0], scale=1, window_width=1080, window_height=720, show=False, save="test.png", flag=0):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(example[:,:3])

    meshes = []
    for i in range(example.shape[0]):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.0125)
        ball.translate(example[i,:3])
        ball.rotate(rotation_matrix(0, np.pi, np.pi), center=np.array([0,0,0]))
        meshes.append(ball)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height, visible=True)
    for ball in meshes:
        vis.add_geometry(ball)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.90, 0.90, 0.90])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.ZCoordinate

    control = vis.get_view_control()
    # control.convert_from_pinhole_camera_parameters(camera_parameters)
    control.rotate(400, 0)
    control.rotate(0, 100)
    if flag:
        control.rotate(0, -50)
    control.scale(6)
    vis.update_geometry(pcd)
    
    if show:
        vis.run()
    elif save is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save)
        vis.destroy_window()

import mitsuba as mi
from pointflow_fig_colorful import colorful_pcd
import subprocess

def draw_one_example_colorful(example, save="test.png"):
    # 设置Mitsuba 3的渲染变体（默认是scalar_rgb）
    mi.set_variant('scalar_rgb')

    # 将点云数据保存为Mitsuba 3的场景文件
    xml_filename = "tmp.xml"
    colorful_pcd(example, xml_filename)  # 假设colorful_pcd函数生成Mitsuba 3兼容的XML文件

    # 加载场景
    scene = mi.load_file(xml_filename)

    # 渲染场景
    image = mi.render(scene)

    # 保存渲染结果
    mi.util.write_bitmap(save, image)

    subprocess.run(["rm", xml_filename])

def draw_examples(tag, examples, colorful=True):

    # os.makedirs("figures/{}".format(tag), exist_ok=True)
    # for i, example in enumerate(examples):
    #     if not colorful:
    #         draw_one_example(example, window_width=720, window_height=600, show=False, save="figures/{}/example_{}.png".format(tag, i))
    #     else:
    #         draw_one_example_colorful(example, save="figures/{}/example_{}.png".format(tag, i))
    
    # matplotlib.rcParams.update({'font.size': 13, 'font.weight': 'bold'})
    
    ###
    if len(examples) == 6:
        fig, axes = plt.subplots(1, 6, figsize=(60, 12))
        for i in range(6):
            ax = axes[i]
            im = Image.open("figures/{}/example_{}.png".format(tag, i))
            w, h = im.size
            im = im.crop((w * 0.15, h * 0.25, w * 0.85, h * 0.95))
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.imshow(im, extent=[0, 1, 0, 1])
            if i == 0:
                ax.set_title('原图', y=0, fontsize=15 * 4)
            else:
                ax.set_title('噪声等级：' + str(i), y=0, fontsize=15 * 4)
            ax.axis('off')
            plt.tight_layout(pad=0, h_pad=0, w_pad=0)
            plt.savefig("figures/{}/{}.png".format(tag, tag))
            plt.savefig("figures/{}/{}.pdf".format(tag, tag))
    ###
    else:
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        for i in range(15):
            ax = axes[i//5][i%5]
            im = Image.open("figures/{}/example_{}.png".format(tag, i))
            w, h = im.size
            im = im.crop((w * 0.15, h * 0.25, w * 0.85, h * 0.95))
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.imshow(im, extent=[0, 1, 0, 1])
            ax.set_title(c_list[i], y=0)
            ax.axis('off')
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig("figures/{}/examples.pdf".format(tag))

examples = []
modelnet = ModelNet40Ply2048(split = 'test')
clean_data, labels = modelnet.data[:, 0:1024, :], modelnet.label
examples.append(clean_data[328])

for index, c in enumerate(c_list): #, [53, 70, 113, 119, 166, 179]: #range(0, len(data)):
    for i in range(5):
        modelnet_c = ModelNet40_C(split = 'test', corruption = c, severity = i + 1)
        data_c, labels_c = modelnet_c.data, modelnet_c.label
        if c == 'rotation':
            examples.append(rotation(examples[0].copy(), severity=i))
        else:
            examples.append(normalize(data_c[328]))

    draw_examples(tag = '328_' + c, examples = examples)

    examples = []
    examples.append(clean_data[328])


# draw_one_example_colorful()