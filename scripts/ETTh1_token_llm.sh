export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

export ROOT_PATH="./data/ETT"
export DATA_PATH="ETTh1.csv"

export SEQ_LEN=96
export PATCH_SIZE=4
export STRIDE=2
export VOCAB_SIZE=128
export D_MODEL=256
export N_LAYERS=4
export N_HEADS=4
export DROPOUT=0.1
export USE_INSTANCE_NORM=true
export DECODE_TEMPERATURE=0.8
export USE_LINEAR_SHORTCUT=true
export GPT_MODEL_NAME="openai-community/gpt2"
export GPT_LOCAL_PATH="./gpt"
export USE_PRETRAINED_GPT2=false
export PREFER_LOCAL_GPT2=true
export GPT_LOCAL_FILES_ONLY=true

export BATCH_SIZE=32
export LEARNING_RATE=0.0001
export WEIGHT_DECAY=0.0001
export TRAIN_EPOCHS=50
export PATIENCE=10
export LRADJ="type3"

export ALPHA=0.1
export BETA=0.1
export GAMMA=0.0
export MAX_GRAD_NORM=1.0

export USE_MULTIVARIATE=false
export TARGET_COL="OT"
export USE_GPU=true
export NUM_WORKERS=0
export FREQ="h"
export SEED=42

for PRED_LEN in 96 ; do
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
    --use_instance_norm $USE_INSTANCE_NORM \
    --decode_temperature $DECODE_TEMPERATURE \
    --use_linear_shortcut $USE_LINEAR_SHORTCUT \
    --gpt_model_name $GPT_MODEL_NAME \
    --gpt_local_path $GPT_LOCAL_PATH \
    --use_pretrained_gpt2 $USE_PRETRAINED_GPT2 \
    --prefer_local_gpt2 $PREFER_LOCAL_GPT2 \
    --gpt_local_files_only $GPT_LOCAL_FILES_ONLY \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --lradj $LRADJ \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $GAMMA \
    --max_grad_norm $MAX_GRAD_NORM \
    --use_multivariate $USE_MULTIVARIATE \
    --target_col $TARGET_COL \
    --use_gpu $USE_GPU \
    --num_workers $NUM_WORKERS \
    --freq $FREQ \
    --seed $SEED
done;
