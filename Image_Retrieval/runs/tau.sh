SAVE_PATH="runs"

## CE+
python3 tau.py \
    2>&1 | tee -a ${SAVE_PATH}/"tau.log"