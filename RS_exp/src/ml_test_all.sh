METHOD_LIST=(RankNet ListNet ListMLE LambdaRank ApproxNDCG NeuralNDCG SONG K-SONG)
for METHOD in ${METHOD_LIST[*]};
do
    echo $METHOD
    python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256  --eval_batch_size 32 \
               --loss_type NDCG \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 0 \
               --train 0 --buffer 0 \
               --test_all 1 \
               --pretrain_model ../model/NeuMF/ml-20m_$METHOD.pt \
               --gpu 0 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name ml-20m_test_all_$METHOD
done

