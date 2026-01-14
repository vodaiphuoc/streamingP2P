export CUDA_VISIBLE_DEVICES="0,1"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

python3 test.py \
    --dst_model my.pt \
    --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
    --num ${num_gpus} \
    --val_best