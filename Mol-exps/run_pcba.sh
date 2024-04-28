# conda activate vlp

DATASET=ogbg-molpcba
GNN_TYPE=gin
#FILENAME=${DATASET}_${GNN_TYPE}_ce-loss_map_at_100.pkl
FILENAME=${DATASET}_${GNN_TYPE}_map-loss_map_at_50.pkl

CUDA_VISIBLE_DEVICES=6 python main_pcba.py --dataset ${DATASET} --gnn ${GNN_TYPE} --filename ${FILENAME}
