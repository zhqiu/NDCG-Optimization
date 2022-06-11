#python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
#               --dropout 0.2 --dataset 'netflix' --batch_size 256  --eval_batch_size 512 \
#               --loss_type RankNet \
#               --metric NDCG \
#               --ndcg_gamma 0.1 --ndcg_topk -1 \
#               --load 1 --init_last 1 \
#               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
#               --gpu 0 --reorg_train_data 1 \
#               --epoch 120 \
#               --random_seed 0 \
#               --run_name netflix_RankNet &

#python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
#               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
#               --loss_type ListNet \
#               --metric NDCG \
#               --ndcg_gamma 0.1 --ndcg_topk -1 \
#               --load 1 --init_last 1 \
#               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
#               --gpu 1 --reorg_train_data 1 \
#               --epoch 120 \
#               --random_seed 0 \
#               --run_name netflix_ListNet &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type ListMLE \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 2 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_ListMLE &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type NeuralNDCG \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 3 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_NeuralNDCG &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256  --eval_batch_size 512 \
               --loss_type ApproxNDCG \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 4 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_ApproxNDCG &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type LambdaRank \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 5 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_LambdaRank &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type NDCG \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk -1 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 6 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_SONG &

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'netflix' --batch_size 256 --eval_batch_size 512 \
               --loss_type NDCG \
               --metric NDCG \
               --ndcg_gamma 0.3 --ndcg_topk 300 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/netflix_pretrained.pt \
               --gpu 7 --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name netflix_K-SONG

