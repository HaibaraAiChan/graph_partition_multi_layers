#!/bin/bash

File=main_dgl_arxiv_gat.py
Data=('ogbn-arxiv')

python python $File \
        --seed 1236 \
        &> logs/gat/${Data}_gat_1236.log






# File=main_dgl_citation_gat.py

# # Data=(cora pubmed)
# Data=(pubmed)
# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed 1235 \
#         &> logs/gat/${data}_gat_1235.log
# done

# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed 1236 \
#         &> logs/gat/${data}_gat_1236.log
# done

# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed 1238 \
#         &> logs/gat/${data}_gat_1238.log
# done

# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         --seed 1237 \
#         &> logs/gat/${data}_gat_1237.log
# done

# File=main_dgl_arxiv_gat.py
# Data=(ogbn-arxiv)

# for data in ${Data[@]}
# do
#     python $File \
#         --dataset $data \
#         &> logs/gat/${data}_gat_1236.log
# done
