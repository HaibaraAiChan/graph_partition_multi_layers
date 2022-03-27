#!/bin/bash

# File=full_batch_arxiv_sage.py
File=full_batch_arxiv_sage_GPU_mem.py
Data=ogbn-arxiv

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
# fan_out=(10,25,10 10,25,15 10,25,20 25,35,40 10,50,100 50,100,200)
fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
# fan_out=(10,25,10)
epoch=1
# Aggre=mean
AggreList=(lstm)
# AggreList=(lstm mean)


for Aggre in ${AggreList[@]}
do
        logPath=../logs/sage/1_runs/pure_train/${Aggre}/
        mkdir $logPath
        rm $logPath* 
        for fan_out in ${fan_out_list[@]}
        do
                python $File \
                --dataset $Data \
                --aggre $Aggre \
                --seed $seed \
                --setseed $setseed \
                --GPUmem $GPUmem \
                --gen-full-batch True \
                --load-full-batch False \
                --lr $lr \
                --num-runs $run \
                --num-epochs $epoch \
                --num-layers $layers \
                --num-hidden $hidden \
                --dropout $dropout \
                --fan-out $fan_out 
                # &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
        done
done

for Aggre in ${AggreList[@]}
do
        logPath=../logs/sage/1_runs/pure_train/${Aggre}/
        # mkdir $logPath
        # rm $logPath* 
        for fan_out in ${fan_out_list[@]}
        do
                python $File \
                --dataset $Data \
                --aggre $Aggre \
                --seed $seed \
                --setseed $setseed \
                --GPUmem $GPUmem \
                --gen-full-batch False \
                --load-full-batch True \
                --lr $lr \
                --num-runs $run \
                --num-epochs $epoch \
                --num-layers $layers \
                --num-hidden $hidden \
                --dropout $dropout \
                --fan-out $fan_out \
                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
        done
done

