gpu_id_a=0
gpu_id_b=1
gpu_id_c=2
gpu_id_d=3
loss_type=NDCG

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256  --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_a} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 4 \
               --run_name ml-20m_${loss_type}_g03_rs4 &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256 --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.5 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_b} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 4 \
               --run_name ml-20m_${loss_type}_g05_rs4 &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256 --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.7 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_c} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 4 \
               --run_name ml-20m_${loss_type}_g07_rs4 &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256 --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.9 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_d} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 4 \
               --run_name ml-20m_${loss_type}_g09_rs4
