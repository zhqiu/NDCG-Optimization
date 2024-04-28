# conda activate vlp

DATASET=ogbg-molhiv
GNN_TYPE=gin
FILENAME=${DATASET}_${GNN_TYPE}_ce-loss-300.pkl

CUDA_VISIBLE_DEVICES=6 python main_hiv.py --dataset ${DATASET} --gnn ${GNN_TYPE} --filename ${FILENAME}
