SAVE_PATH="runs"
root="/home/panzx/retrieval/HubnessOT/Image_Retrieval/runs/querybank"

python debug.py \
    --dataset=f30k \
    --qemb_path=${root}/f30k_test_txts.npy \
    --temb_path=${root}/f30k_test_imgs.npy \
    --gqemb_path=${root}/llava_f30k_test.npy \
    --gtemb_path=${root}/f30k_dev_imgs.npy \
    2>&1 | tee -a ${SAVE_PATH}/"debug.log"

exit