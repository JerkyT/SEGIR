Str=("add_global" "add_local" "dropout_global" "dropout_local" "jitter" "rotate" "scale")
for i in ${Str[@]}
do
for j in $(seq 0 4)
do
CUDA_VISIBLE_DEVICES=2 python ./examples/classification/main.py \
                            --cfg ./cfgs/PointCloudC/pointnet++.yaml \
                            --dataset.common.corruption $i \
                            --dataset.common.severity $j \
                            --pretrained_path './PointMetaBase/log/modelnet40ply2048/Geometric_SHT_pointnet2/checkpoint/modelnet40ply2048-train-pointnet++-ngpus1-seed9333-20240820-214359-AnPV7C4EEtPWGjm3YNWsob_ckpt_best.pth';
done
done

# CUDA_VISIBLE_DEVICES=3 python ./examples/classification/main.py --cfg ./cfgs/modelnet40ply2048/apes_global.yaml
# apes_global pointmetabase-s apes_local pointnet++