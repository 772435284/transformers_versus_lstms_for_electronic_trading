export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CNN_LSTM" ]; then
    mkdir ./logs/CNN_LSTM
fi


python -u run.py \
  --is_training 1 \
  --root_path /hy-tmp/ \
  --data_path crypto.csv \
  --model_id LOB_100_100 \
  --model CNN_LSTM \
  --freq h \
  --data custom \
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
  --batch_size 256 >logs/CNN_LSTM/CNN_LSTM_diff'_LOB_'100.log
