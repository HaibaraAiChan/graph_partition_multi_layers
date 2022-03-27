#!/bin/bash


run=1

# File=full_graph_train_citation_gat.py
# seed=1238
# Data=(cora)
# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed $seed \
#         --num-runs $run \
#         &> logs/gat/${data}_${seed}.log
# done
# Data=(pubmed)
# seed=1237
# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed $seed \
#         --num-runs $run \
#         &> logs/gat/${data}_${seed}.log
# done

# python full_graph_train_arxiv_gat.py --seed 1236 --num-runs 1 &> logs/gat/arxiv_1236.log

# File=full_graph_train_reddit_gat.py
# Data=(reddit)
# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed 1236 \
#         --num-runs $run \
#         &> logs/gat/${data}_1236.log
# done


# python full_graph_train_citation_sage.py --dataset cora --seed 1236 --num-runs 1 --num-epochs 30 &> logs/sage/cora_1236.log
# python full_graph_train_citation_sage.py --dataset pubmed --seed 1238 --num-runs 1 --num-epochs 30 &> logs/sage/pubmed_1238.log
python full_graph_train_arxiv_sage.py --seed 1236  --num-runs 1 --num-epochs 30 &> logs/sage/lstm_arxiv_1236.log
# python full_graph_train_products_sage.py --seed 1236 --num-runs 1 --num-epochs 30 &> logs/sage/products_1236.log
# python full_graph_train_reddit_sage.py --seed 1236 --num-runs 1 --num-epochs 30 &> logs/sage/reddit_1236.log
