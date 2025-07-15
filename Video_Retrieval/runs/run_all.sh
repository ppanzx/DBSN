root="runs"
SAVE_PATH=$root

## CE+
python3 sn.py \
    --sims_path=$root/sims/ce/didemo-ce+/didemo_sim.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

python3 sn.py \
    --sims_path=$root/sims/ce/msrvtt-ce+/msrvtt_sim.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

## TT-CE+
python3 sn.py \
    --sims_path=$root/sims/ce/didemo-tt_ce+/didemo_sim.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"


python3 sn.py \
    --sims_path=$root/sims/ce/msrvtt-tt_ce+/msrvtt_sim.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"


## CLIP4CLIP
python3 sn.py \
    --sims_path=$root/sims/clip4clip/didemo-meanP/clip4clip_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

python3 sn.py \
    --sims_path=$root/sims/clip4clip/msrvtt-meanP/clip4clip_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"


## DRL
python3 sn.py \
    --sims_path=$root/sims/drl/didemo/drl_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

python3 sn.py \
    --sims_path=$root/sims/drl/msrvtt/drl_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

## Xpool
python3 sn.py \
    --sims_path=$root/sims/xpool/didemo/xpool_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"

python3 sn.py \
    --sims_path=$root/sims/xpool/msrvtt/xpool_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn_all.log"
