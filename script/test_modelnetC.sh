Str=("background" "cutout" "density" "density_inc" "distortion" "distortion_rbf" "distortion_rbf_inv" "gaussian" "impulse" "rotation" "shear" "uniform" "upsampling"  "occlusion" "lidar")
# Str=("occlusion" "lidar")
for i in ${Str[@]}
do
for j in $(seq 1 5)
do
CUDA_VISIBLE_DEVICES=0 python ./examples/classification/main.py \
                            --cfg ./cfgs/PointCloudC/pointnet++.yaml \
                            --dataset.common.corruption $i \
                            --dataset.common.severity $j \
                            --pretrained_path './PointMetaBase/log/modelnet40ply2048/Cut_SapceT_KD_pointnet2/checkpoint/modelnet40ply2048-train-pointnet++-ngpus1-seed9333-20250420-234604-iHtvtxMQNMzjW2NMmrxVJH_ckpt_best.pth';
done
done

# CUDA_VISIBLE_DEVICES=0 python ./examples/classification/main.py --cfg ./cfgs/modelnet40ply2048/pointnet++.yaml
# apes_global pointmetabase-s apes_local pointnet++