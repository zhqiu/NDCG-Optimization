gpu_id_a=3
gpu_id_b=4
gpu_id_c=5
loss_type=NDCG
py=python
des=no_warmup

${py} main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256  --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk -1 \
               --load 0 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu ${gpu_id_a} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_${loss_type}_${des}_rs0 &

${py} main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk -1 \
               --load 0 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu ${gpu_id_b} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 2 \
               --run_name netflix_${loss_type}_${des}_rs2 &

${py} main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk -1 \
               --load 0 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu ${gpu_id_c} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 4 \
               --run_name netflix_${loss_type}_${des}_rs4
