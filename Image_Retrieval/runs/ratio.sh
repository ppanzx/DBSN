SAVE_PATH="runs"

## CE+
python3 ratio.py \
    2>&1 | tee -a ${SAVE_PATH}/"ratio.log.1"