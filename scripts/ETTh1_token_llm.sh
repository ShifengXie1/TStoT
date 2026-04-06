export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

export ROOT_PATH="./data/ETT"
export DATA_PATH="ETTh1.csv"

export SEQ_LEN=96
export PATCH_SIZE=16
export STRIDE=16
export VOCAB_SIZE=512
export D_MODEL=128
export N_LAYERS=4
export N_HEADS=4
export DROPOUT=0.1

export BATCH_SIZE=32
export LEARNING_RATE=0.001
export TRAIN_EPOCHS=10
export PATIENCE=3
export LRADJ="type3"

export ALPHA=0.5
export BETA=0.5
export GAMMA=0.1

export USE_MULTIVARIATE=false
export TARGET_COL="OT"
export USE_GPU=true
export NUM_WORKERS=0
export FREQ="h"
export SEED=42

for PRED_LEN in 96 192 336 720; do
    python -u run.py \
    --is_training 1 \
    --model token_llm_forecasting \
    --data ETTh1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --patch_size $PATCH_SIZE \
    --stride $STRIDE \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --lradj $LRADJ \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $GAMMA \
    --use_multivariate $USE_MULTIVARIATE \
    --target_col $TARGET_COL \
    --use_gpu $USE_GPU \
    --num_workers $NUM_WORKERS \
    --freq $FREQ \
    --seed $SEED
done;
