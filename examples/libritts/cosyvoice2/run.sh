#!/bin/bash
. ./path.sh || exit 1;

OPTS=$(getopt -o "" --long hf_repo:,token: -- "$@")
eval set -- "$OPTS"

stage=0 # Start from 0 since you already have your data
stop_stage=8

# EDIT THESE PATHS
hf_repo=$2
token=$4
pretrained_model_dir="../../../pretrained_models/CosyVoice2-0.5B"


# Stage 0: Preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation for Vietnamese HF Dataset"

  for x in train dev; do
    echo data/$x
    python local/prepare_hf_data.py \
        --input_dir $hf_repo \
        --des_dir data/$x \
        --token $token
  done
  
fi

# Stage 1: Speaker Embedding
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extracting CAM++ embeddings"
  for x in train dev; do
    python tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# Stage 2: Speech Tokens
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extracting speech tokens"
  for x in train dev; do
    python tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

# Stage 3: Parquet Data
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making Parquet files"
  for x in train dev; do
    mkdir -p data/$x/parquet
    python tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 --src_dir data/$x --des_dir data/$x/parquet
  done
fi

# Stage 5: Training
# (Update the 'cat' commands to point to your new folders)
export CUDA_VISIBLE_DEVICES="0,1"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  cp data/train/parquet/data.list data/train.data.list
  cp data/dev/parquet/data.list data/dev.data.list
  
  for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
