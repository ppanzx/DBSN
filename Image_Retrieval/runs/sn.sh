# python sn.py \
#     --dataset=f30k \
#     --sims_path=runs/sims/zs_f30k_sims.npy

#   --sims_path=runs/sims/zs_coco_sims.npy \

## sims
# python dbsn.py \
#     --dataset=f30k \
#     --sims_path=runs/sims/zs_f30k_sims.npy \
#     --qtsims_path=runs/sims/f30k_train_txts_test_imgs_sims.npy \
#     --qqsims_path=runs/sims/f30k_train_txts_train_imgs_sims.npy 


# python dbsn.py \
#     --dataset=f30k \
#     --qemb_path=runs/querybank/f30k_test_txts.npy \
#     --temb_path=runs/querybank/f30k_test_imgs.npy \
#     --gqemb_path=runs/querybank/f30k_train_txts.npy \
#     --gtemb_path=runs/querybank/f30k_train_imgs.npy 

# python dbsn.py \
#     --dataset=f30k \
#     --qemb_path=runs/querybank/f30k_test_txts.npy \
#     --temb_path=runs/querybank/f30k_test_imgs.npy \
#     --gqemb_path=runs/querybank/f30k_dev_txts.npy \
#     --gtemb_path=runs/querybank/f30k_dev_imgs.npy 

# python dbsn.py \
#     --dataset=f30k \
#     --qemb_path=runs/querybank/f30k_test_txts.npy \
#     --temb_path=runs/querybank/f30k_test_imgs.npy \
#     --gqemb_path=runs/querybank/f30k_test_txts.npy \
#     --gtemb_path=runs/querybank/f30k_test_imgs.npy 

python sn.py \
    --sims_path=runs/sims/zs_f30k_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/zs_coco_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/ft_f30k_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/ft_coco_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/albef_f30k_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"

python sn.py \
    --sims_path=runs/sims/albef_coco_sims.npy \
    2>&1 | tee -a ${SAVE_PATH}/"sn.log"