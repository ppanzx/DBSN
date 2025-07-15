SAVE_PATH="runs"
root="/home/panzx/retrieval/LoraCLIP/itr/scripts/runs/zs_clip_coco"

python dbsn.py \
    --dataset=coco \
    --qemb_path=${root}/coco_testall_txts.npy \
    --temb_path=${root}/coco_testall_imgs.npy \
    --gqemb_path=${root}/coco_train_txts_100k.npy \
    --gtemb_path=${root}/coco_train_imgs_20k.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

exit

python dbsn.py \
    --dataset=coco \
    --qemb_path=${root}/coco_testall_txts.npy \
    --temb_path=${root}/coco_testall_imgs.npy \
    --gqemb_path=${root}/coco_dev_txts.npy \
    --gtemb_path=${root}/coco_dev_imgs.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

######################################################################
root="/home/panzx/retrieval/LoraCLIP/itr/scripts/runs/ft_clip_f30k"
python dbsn.py \
    --dataset=f30k \
    --qemb_path=${root}/f30k_test_txts.npy \
    --temb_path=${root}/f30k_test_imgs.npy \
    --gqemb_path=${root}/f30k_dev_txts.npy \
    --gtemb_path=${root}/f30k_dev_imgs.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

python dbsn.py \
    --dataset=f30k \
    --qemb_path=${root}/f30k_test_txts.npy \
    --temb_path=${root}/f30k_test_imgs.npy \
    --gqemb_path=${root}/f30k_train_txts.npy \
    --gtemb_path=${root}/f30k_train_imgs.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

######################################################################
root="/home/panzx/retrieval/LoraCLIP/itr/scripts/runs/ft_clip_coco"
python dbsn.py \
    --dataset=coco \
    --qemb_path=${root}/coco_testall_txts.npy \
    --temb_path=${root}/coco_testall_imgs.npy \
    --gqemb_path=${root}/coco_train_txts_100k.npy \
    --gtemb_path=${root}/coco_train_imgs_20k.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

python dbsn.py \
    --dataset=coco \
    --qemb_path=${root}/coco_testall_txts.npy \
    --temb_path=${root}/coco_testall_imgs.npy \
    --gqemb_path=${root}/coco_train_txts_50k.npy \
    --gtemb_path=${root}/coco_train_imgs_10k.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"

python dbsn.py \
    --dataset=coco \
    --qemb_path=${root}/coco_testall_txts.npy \
    --temb_path=${root}/coco_testall_imgs.npy \
    --gqemb_path=${root}/coco_dev_txts.npy \
    --gtemb_path=${root}/coco_dev_imgs.npy \
    2>&1 | tee -a ${SAVE_PATH}/"dbsn.log"
######################################################################
