export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Transformer" ]; then
    mkdir ./logs/Transformer
fi


python -u run.py \
  --is_training 0 \
  --root_path /hy-tmp/ \
  --data_path crypto.csv \
  --model_id LOB_100_100 \
  --model Transformer \
  --freq h \
  --data custom \
  --checkpoints /hy-tmp/models_diff/ \
  --target mid_price \
  --features MS \
  --seq_len 100 \
  --label_len 50 \
  --pred_len 100 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 41 \
  --dec_in 41 \
  --c_out 1 \
  --des 'exp' \
  --itr 1 \
  --batch_size 256 >logs/Transformer/Transformer_diff'_LOB_'100.log
