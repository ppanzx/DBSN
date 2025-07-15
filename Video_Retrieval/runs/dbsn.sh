# python dbsn.py \
#     --dataset=f30k \
#     --qemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_test_txts.npy \
#     --temb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_test_vids.npy \
#     --gqemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_train_txts.npy \
#     --gtemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_train_vids.npy

# python dbsn.py \
#     --dataset=f30k \
#     --qemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_test_txts.npy \
#     --temb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_test_vids.npy \
#     --gqemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_jsfusion_additional_txts.npy \
#     --gtemb_path=runs/embs/clip4clip_msrvtt/CLIP4Clip_msrvtt_train_vids.npy

        # --gqemb_path=runs/embs/clip4clip_f30k/CLIP4Clip_caps_train_txts.npy \

# root=/data2/panzhengxin/retrieval/LoraDRL/ckpts/msrvtt_clip4clip
# python dbsn.py \
#     --dataset=f30k \
#     --sims_path=$root/CLIP4Clip_q_t_sims.npy \
#     --qtsims_path=$root/CLIP4Clip_gq_t_sims.npy \
#     --gqgt_sims_path=$root/CLIP4Clip_gq_gt_sims.npy
SAVE_PATH="runs"

root=/home/panzx/retrieval/LoraDRL/ckpts/aaai2025/ckpt_msrvtt_ctf/train_test
python dbsn.py \
    --dataset=f30k \
    --sims_path=$root/q_t_sims.npy \
    --qtsims_path=$root/gq_t_sims.npy \
    --gqgt_sims_path=$root/gq_gt_sims.npy \
    --gttsims_path=$root/gt_t_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

exit
root=/home/panzx/retrieval/LoraDRL/ckpts/aaai2025/didemo_clip4clip/val_test
python dbsn.py \
    --dataset=f30k \
    --sims_path=$root/q_t_sims.npy \
    --qtsims_path=$root/gq_t_sims.npy \
    --gqgt_sims_path=$root/gq_gt_sims.npy \
    --gttsims_path=$root/gt_t_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

root=/home/panzx/retrieval/LoraDRL/ckpts/aaai2025/didemo_clip4clip/train_test
python dbsn.py \
    --dataset=f30k \
    --sims_path=$root/q_t_sims.npy \
    --qtsims_path=$root/gq_t_sims.npy \
    --gqgt_sims_path=$root/gq_gt_sims.npy \
    --gttsims_path=$root/gt_t_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"


    
exit

