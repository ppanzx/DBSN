export CUDA_VISIBLE_DEVICES=1
SAVE_PATH="runs"

python complexity.py --dataset=f30k --split=test --batch_size=5000 \
    2>&1 | tee -a ${SAVE_PATH}/"time.log"
exit