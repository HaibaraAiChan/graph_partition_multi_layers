#!/bin/bash

File=full_batch_arxiv_sage.py
Data=ogbn-arxiv

Aggre=mean
model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3
# batch_size=(45471)
hidden=256
run=1
# fan_out=(10,25,10 10,25,15 10,25,20)
fan_out=(25,35,40 10,50,100 50,100,200)
# fan_out=(10,25,10)
epoch=30
for fan_out in ${fan_out[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --setseed $setseed \
        --GPUmem $GPUmem \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out \
        --eval &> ../logs/sage/1_runs/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log

done
