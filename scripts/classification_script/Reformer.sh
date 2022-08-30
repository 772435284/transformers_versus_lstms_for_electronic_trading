export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Reformer" ]; then
    mkdir ./logs/Reformer
fi

python -u run.py \
  --is_training 1 \
  --root_path /hy-tmp/ \
  --data_path eth_usdt_label.csv \
  --product eth \
  --model_id LOB \
  --model Reformer \
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
  --horizon label_5 \
  --label_method 2 \
  --batch_size 64 >logs/Reformer/Reformer_LOB_label_5_product_eth_method_2_.log 


python -u run.py \
  --is_training 1 \
  --root_path /hy-tmp/ \
  --data_path eth_usdt_label.csv \
  --product eth \
  --model_id LOB \
  --model Reformer \
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
  --horizon label_4 \
  --label_method 2 \
  --batch_size 64 >logs/Reformer/Reformer_LOB_label_4_product_eth_method_2_.log


python -u run.py \
  --is_training 1 \
  --root_path /hy-tmp/ \
  --data_path eth_usdt_label.csv \
  --product eth \
  --model_id LOB \
  --model Reformer \
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
  --horizon label_3 \
  --label_method 2 \
  --batch_size 64 >logs/Reformer/Reformer_LOB_label_3_product_eth_method_2_.log


python -u run.py \
  --is_training 1 \
  --root_path /hy-tmp/ \
  --data_path eth_usdt_label.csv \
  --product eth \
  --model_id LOB \
  --model Reformer \
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
  --horizon label_2 \
  --label_method 2 \
  --batch_size 64 >logs/Reformer/Reformer_LOB_label_2_product_eth_method_2_.log










