gpu_id_a=7
loss_type=NeuralNDCG

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256  --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --neuralndcg_temp 0.2 \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_a} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name ml-20m_${loss_type}_hyper_tune_neuralndcg_temp_02

