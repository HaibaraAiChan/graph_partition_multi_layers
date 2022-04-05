#!/bin/bash



File=full_batch_arxiv_sage.py
# File=full_batch_arxiv_sage_GPU_mem.py
Data=ogbn-arxiv
savePath=../logs/sage/1_runs/pure_train/ogbn_arxiv
mkdir $savePath
model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3

hidden=256
run=1

fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
# fan_out=(10,25,10)
# epoch=30
# Aggre=mean
# AggreList=(lstm)
# AggreList=(lstm mean)

epoch=2

hiddenList=(32 64 128 256)
layersList=(3)
# layersList=(3 4 5 6)
fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
AggreList=(mean lstm)



for Aggre in ${AggreList[@]}
do
        mkdir ${savePath}/${Aggre}/
        for layers in ${layersList[@]}
        do      
                mkdir ${savePath}/${Aggre}/layers_${layers}/
                
                for hidden in ${hiddenList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/
                        echo ${savePath}/${Aggre}/layers_${layers}/h_${hidden}
                        for fan_out in ${fan_out_list[@]}
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
                                --log-indent 1 \
                                &> ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
                        done
                done
        done
done

layersList=(4)
# layersList=(3 4 5 6)
fan_out_list=(25,35,40,40)

for Aggre in ${AggreList[@]}
do
        mkdir ${savePath}/${Aggre}/
        for layers in ${layersList[@]}
        do      
                mkdir ${savePath}/${Aggre}/layers_${layers}/
                
                for hidden in ${hiddenList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/
                        echo ${savePath}/${Aggre}/layers_${layers}/h_${hidden}
                        for fan_out in ${fan_out_list[@]}
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
                                --log-indent 1 \
                                &> ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
                        done
                done
        done
done

layersList=(5)
# layersList=(3 4 5 6)
fan_out_list=(25,35,40,40,40)

for Aggre in ${AggreList[@]}
do
        mkdir ${savePath}/${Aggre}/
        for layers in ${layersList[@]}
        do      
                mkdir ${savePath}/${Aggre}/layers_${layers}/
                
                for hidden in ${hiddenList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/
                        echo ${savePath}/${Aggre}/layers_${layers}/h_${hidden}
                        for fan_out in ${fan_out_list[@]}
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
                                --log-indent 1 \
                                &> ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
                        done
                done
        done
done
layersList=(6)
# layersList=(3 4 5 6)
fan_out_list=(25,35,40,40,40,40)

for Aggre in ${AggreList[@]}
do
        mkdir ${savePath}/${Aggre}/
        for layers in ${layersList[@]}
        do      
                mkdir ${savePath}/${Aggre}/layers_${layers}/
                
                for hidden in ${hiddenList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/
                        echo ${savePath}/${Aggre}/layers_${layers}/h_${hidden}
                        for fan_out in ${fan_out_list[@]}
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
                                --log-indent 1 \
                                &> ${savePath}/${Aggre}/layers_${layers}/h_${hidden}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_r_${run}_ep_${epoch}.log
                        done
                done
        done
done