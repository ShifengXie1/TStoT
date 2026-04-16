export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20

export ROOT_PATH="./data/ETT"
export DATA_PATH="ETTh1.csv"

export SEQ_LEN=96
export PATCH_SIZE=8
export STRIDE=2
export VOCAB_SIZE=128
export D_MODEL=256
export N_LAYERS=4
export N_HEADS=4
export DROPOUT=0.1
export USE_LINEAR_SHORTCUT=true
export GPT_MODEL_NAME="openai-community/gpt2"
export GPT_LOCAL_PATH="./gpt"
export USE_PRETRAINED_GPT2=true
export PREFER_LOCAL_GPT2=true
export GPT_LOCAL_FILES_ONLY=true

# CT-GPT2 specific options
# Updated from the old token_llm_forecasting script:
# - model -> ct_gpt2
# - tokenization-specific runtime flags removed from python call
# - added continuous-decoding / probabilistic forecasting options below
export USE_CHRONOS_SCALING=false
export SCALING_EPS=1e-8
export DECODER_HIDDEN_DIM=256
export NUM_OUTPUT_MIXTURES=1
export NUM_SAMPLING_PATHS=0
export MIN_LOG_VARIANCE=-10.0
export MAX_LOG_VARIANCE=5.0
export USE_ALIGNMENT=false
export ALIGNMENT_HIDDEN_DIM=256
export CONTRASTIVE_TEMPERATURE=0.1

export BATCH_SIZE=32
export LEARNING_RATE=0.0001
export WEIGHT_DECAY=0.0001
export TRAIN_EPOCHS=50
export PATIENCE=3
export LRADJ="type3"

export LAMBDA_PRED=1.0
export LAMBDA_CON=0.0
export LAMBDA_TREND=0.0
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
    --model ct_gpt2 \
    --data ETTh1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --use_linear_shortcut $USE_LINEAR_SHORTCUT \
    --use_chronos_scaling $USE_CHRONOS_SCALING \
    --scaling_eps $SCALING_EPS \
    --gpt_model_name $GPT_MODEL_NAME \
    --gpt_local_path $GPT_LOCAL_PATH \
    --use_pretrained_gpt2 $USE_PRETRAINED_GPT2 \
    --prefer_local_gpt2 $PREFER_LOCAL_GPT2 \
    --gpt_local_files_only $GPT_LOCAL_FILES_ONLY \
    --decoder_hidden_dim $DECODER_HIDDEN_DIM \
    --num_output_mixtures $NUM_OUTPUT_MIXTURES \
    --num_sampling_paths $NUM_SAMPLING_PATHS \
    --min_log_variance $MIN_LOG_VARIANCE \
    --max_log_variance $MAX_LOG_VARIANCE \
    --use_alignment $USE_ALIGNMENT \
    --alignment_hidden_dim $ALIGNMENT_HIDDEN_DIM \
    --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --lradj $LRADJ \
    --lambda_pred $LAMBDA_PRED \
    --lambda_con $LAMBDA_CON \
    --lambda_trend $LAMBDA_TREND \
    --max_grad_norm $MAX_GRAD_NORM \
    --use_multivariate $USE_MULTIVARIATE \
    --target_col $TARGET_COL \
    --use_gpu $USE_GPU \
    --num_workers $NUM_WORKERS \
    --freq $FREQ \
    --seed $SEED
done;
