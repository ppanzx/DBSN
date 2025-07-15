SAVE_PATH="runs"

python skewness.py \
    --dataset=f30k \
    --sims_path=runs/sims/zs_f30k_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"skewness.log"

python skewness.py \
    --dataset=coco \
    --sims_path=runs/sims/ft_coco_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"skewness.log"