export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FEDformer" ]; then
    mkdir ./logs/FEDformer
fi

for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 0 \
  --root_path /hy-tmp/ \
  --data_path btc_reg.csv \
  --model_id LOB_96_$pred_len \
  --model FEDformer \
  --freq us \
  --data custom \
  --checkpoints /hy-tmp/models_price/ \
  --target mid_price \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 41 \
  --dec_in 41 \
  --c_out 1 \
  --des 'exp' \
  --itr 1 \
  --batch_size 32 >logs/FEDformer/FEDformer'_LOB_'$pred_len.log
done