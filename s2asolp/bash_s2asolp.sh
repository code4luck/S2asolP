#!/bin/bash
export PYTHONPATH="$PYTHONPATH:./"

DATASET="deepsol"
BATCH_SIZE=4
MODEL_PATH="model/saprot_pdb"
TOKENIZER="model/saprot_pdb"
DEVICES=1
NUM_NODES=1
MAX_EPOCHS=40
ACC_BATCH=1
LR=1e-4
PATIENCE=10
STRATEGY="auto"
FINETUNE="head"
POOLING_HEAD="attention1d"
FILE_NAME="s2solp"
wandb_proj="s2solp"
SEQ_MAX_LEN=1200

TRAIN_DATA="data/saprot_train.csv"
VAL_DATA="data/saprot_val.csv"
TEST_DATA="data/saprot_test.csv"
TRAIN_BIO="data/bio_train.pkl"
VAL_BIO="data/bio_val.pkl"
TEST_BIO="data/bio_test.pkl"

python src/train.py \
  --dataset $DATASET \
  --batch_size $BATCH_SIZE \
  --model_path $MODEL_PATH \
  --tokenizer $TOKENIZER \
  --pooling_head $POOLING_HEAD \
  --file_name $FILE_NAME \
  --devices $DEVICES \
  --strategy $STRATEGY \
  --num_nodes $NUM_NODES \
  --max_epochs $MAX_EPOCHS \
  --accumulate_grad_batches $ACC_BATCH \
  --optim_lr $LR \
  --patience $PATIENCE \
  --optim_finetune $FINETUNE \
  --wandb_project $wandb_proj \
  --seq_max_length $SEQ_MAX_LEN \
  --data_path $TRAIN_DATA $VAL_DATA $TEST_DATA \
  --bio_feature_paths $TRAIN_BIO $VAL_BIO $TEST_BIO \
 

