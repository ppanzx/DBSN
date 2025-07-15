export CUDA_VISIBLE_DEVICES=1
SAVE_PATH="runs"

python zero_shot_retrieval.py --dataset=f30k --split=test \
    2>&1 | tee -a ${SAVE_PATH}/"time.log"
exit


python zero_shot_retrieval.py --dataset=coco --split=testall 
exit

# python zero_shot_retrieval.py --dataset=coco

