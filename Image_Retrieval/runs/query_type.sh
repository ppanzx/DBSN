SAVE_PATH="runs"

## CE+
python3 query_type.py \
    2>&1 | tee -a ${SAVE_PATH}/"query_type.log"