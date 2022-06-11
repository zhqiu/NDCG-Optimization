python main.py --model_name NeuMF --emb_size 64 --lr 0.001 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256 --eval_batch_size 512 --layers '[64]' \
               --loss_type Listwise_CE --warmup_gamma 0.1 \
               --metric NDCG \
               --gpu 1 --reorg_train_data 1 \
               --early_stop -1 \
               --epoch 20 \
               --random_seed 0 \
               --run_name ml-20m_warmup

