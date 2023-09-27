# C + T -> S
python evaluate.py --model=weights/GAFlow-CT2S.pth  --dataset=sintel --global_flow --mixed_precision
# C + T -> K
python evaluate.py --model=weights/GAFlow-CT2K.pth  --dataset=kitti --mixed_precision