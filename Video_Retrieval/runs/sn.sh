SAVE_PATH="runs"

python sn.py \
    --sims_path=runs/sims/clip4clip/msrvtt-meanP/clip4clip_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/clip4clip/didemo-meanP/clip4clip_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/drl/msrvtt/drl_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"
    
python sn.py \
    --sims_path=runs/sims/drl/didemo/drl_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"    

python sn.py \
    --sims_path=runs/sims/xpool/msrvtt/xpool_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/xpool/didemo/didemo_xpool.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

# python sn.py \
#     --dataset=f30k \
#     --sims_path=runs/sims/zs_f30k_sims.npy \
#     2>&1 | tee -a ${SAVE_PATH}/"sn.log"

#   --sims_path=runs/sims/zs_coco_sims.npy \
