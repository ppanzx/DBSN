export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

python zero_shot_retrieval.py --dataset=AudioCaps
python zero_shot_retrieval.py --dataset=Clotho