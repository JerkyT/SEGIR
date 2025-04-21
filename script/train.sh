CUDA_VISIBLE_DEVICES=0 python ./examples/classification/main.py \
                            --cfg ./cfgs/modelnet40ply2048/pointnet++.yaml \
                            --criterion_args.NAME JGEKD \
                            --datatransforms.c TND PointsToTensor PointCloudScaleAndTranslate Spatial_geometry_Enhancement Graph_Domain_Geometry_Enhancement \
                            --mode train \

# Spatial_geometry_Enhancement and Graph_Domain_Geometry_Enhancement are data enhancements based on point cloud skeletons
# criterion_args.NAME uses DKD, which may cause gradient explosion