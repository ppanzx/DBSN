export CUDA_VISIBLE_DEVICES=2

# python extractor.py --dataset=f30k --split=dev --save_path="./runs" --batch_size=1024
# python extractor.py --dataset=f30k --split=test --save_path="./runs" --batch_size=1024

# python extractor.py --dataset=f30k --split=train --save_path="./runs" --batch_size=1024
python extractor.py --dataset=coco --split=dev --save_path="./runs" --batch_size=1024