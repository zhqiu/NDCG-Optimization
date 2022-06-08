gpu_id_a=0
gpu_id_b=1
gpu_id_c=2
loss_type=NDCG_M
des=_
dataset=yahoo

CUDA_VISIBLE_DEVICES=$gpu_id_a python allrank/main.py \
    --config-file-name ${loss_type}_${dataset}.json \
    --run-id ${loss_type}_${dataset}_rs0 \
    --random-seed 0 \
    --job-dir ${loss_type}_${dataset}${des} \
    --pretrained-model './yahoo_pretrained_model.pkl'  &

CUDA_VISIBLE_DEVICES=$gpu_id_b python allrank/main.py \
    --config-file-name ${loss_type}_${dataset}.json \
    --run-id ${loss_type}_${dataset}_rs2 \
    --random-seed 2 \
    --job-dir ${loss_type}_${dataset}${des} \
    --pretrained-model './yahoo_pretrained_model.pkl'  &

CUDA_VISIBLE_DEVICES=$gpu_id_c python allrank/main.py \
    --config-file-name ${loss_type}_${dataset}.json \
    --run-id ${loss_type}_${dataset}_rs4 \
    --random-seed 4 \
    --job-dir ${loss_type}_${dataset}${des} \
    --pretrained-model './yahoo_pretrained_model.pkl'

