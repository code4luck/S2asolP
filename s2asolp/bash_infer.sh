#!/bin/bash
export PYTHONPATH="$PYTHONPATH:./"

DATASET="deepsol"
BATCH_SIZE=4
MODEL_PATH="model/saprot_pdb"
TOKENIZER="model/saprot_pdb"
POOLING_HEAD="attention1d"
SEED=3407
SEQ_MAX_LEN=1200
TRAIN_DATA="data/data_test.csv"
VAL_DATA="data/data_val.csv"
TEST_DATA="data/data_test.csv"
TRAIN_BIO="data/bio_train.pkl"
VAL_BIO="data/bio_val.pkl"
TEST_BIO="data/bio_test.pkl"

python ./infer.py \
  --dataset $DATASET \
  --batch_size $BATCH_SIZE \
  --model_path $MODEL_PATH \
  --tokenizer $TOKENIZER \
  --pooling_head $POOLING_HEAD \
  --seed $SEED \
  --seq_max_length $SEQ_MAX_LEN \
  --data_path $TRAIN_DATA $VAL_DATA $TEST_DATA \
  --bio_feature_paths $TRAIN_BIO $VAL_BIO $TEST_BIO \
 

