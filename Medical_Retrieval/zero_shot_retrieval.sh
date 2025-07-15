export CUDA_VISIBLE_DEVICES=1

python zero_shot_retrieval.py --dataset=books_set

python zero_shot_retrieval.py --dataset=pubmed_set