export CUDA_VISIBLE_DEVICES=0

python zeroshot_resnet_cls.py --model=resnet50 --split=val --save_path="./runs"

python zeroshot_clip_cls.py --model=ViT-B/32 --split=val --save_path="./runs"



