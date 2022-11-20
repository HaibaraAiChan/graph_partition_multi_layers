#!/bin/bash

File=full_batch_products_sage_GPU_mem.py
# File=full_batch_bucket.py
# an_out=10
# python $File --fan-out=$fan_out --num-layers=1
# data=cora
# data=reddit
# data=ogbn-arxiv
# fan_out=10,25,30,40 
# python $File --fan-out=$fan_out --num-layers=4 --num-epochs=1 --num-hidden=1 --dataset=$data

# data=reddit
# fan_out=10,25,30,40 
# python $File --fan-out=$fan_out --num-layers=4 --num-epochs=1 --num-hidden=1 --dataset=$data
data=ogbn-products
# data=karate
# data=ogbn-arxiv
# data=cora
# data=pubmed
# data=reddit

# fan_out=1
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=2
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=5 --num-hidden=1 --dataset=$data

# fan_out=3
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=5 --num-hidden=1 --dataset=$data

# fan_out=4
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=5
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=6
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=7
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=8
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=9
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

fan_out=10
python $File --fan-out=$fan_out --num-layers=1 --num-epochs=10 --num-hidden=1 --dataset=$data
# fan_out=11
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=12
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=15
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=20
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=30
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=50
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=100
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=200
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=300
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=400
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=500
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data
# fan_out=800
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=1 --num-hidden=1 --dataset=$data

# data=pubmed
# fan_out=10,25,30,40,50
# python $File --fan-out=$fan_out --num-layers=5 --num-epochs=1 --num-hidden=1 --dataset=$data

# fan_out=2,4
# python $File --fan-out=$fan_out --num-layers=2 --num-epochs=1 --num-hidden=1 --dataset=$data


# fan_out=10,25
# python $File --fan-out=$fan_out --num-layers=2 --num-epochs=5 --num-hidden=1 --dataset=$data

# fan_out=10,25,30
# python $File --fan-out=$fan_out --num-layers=3 --num-epochs=3

# fan_out=10,25,30,40
# python $File --fan-out=$fan_out --num-layers=4 --num-epochs=3

# fan_out=10,25,30,40,50
# python $File --fan-out=$fan_out --num-layers=5 --num-epochs=3

# fan_out=10,15,20,20,20,20
# python $File --fan-out=$fan_out --num-layers=6
