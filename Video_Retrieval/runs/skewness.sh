SAVE_PATH="runs"

python skewness.py \
    --dataset=f30k \
    --sims_path=./runs/sims/clip4clip/msrvtt-meanP/clip4clip_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"skewness.log"

python skewness.py \
    --dataset=coco \
    --sims_path=./runs/sims/xpool/didemo/didemo_xpool_47.0_44.2.npy \
    2>&1 | tee -a ${SAVE_PATH}/"skewness.log"