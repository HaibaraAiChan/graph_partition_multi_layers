#!/bin/bash

# folderPath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/SAGE

File=pseudo_mini_batch_arxiv_sage.py
Data=ogbn-arxiv

savePath=../logs/sage/1_runs/pure_train/ogbn_arxiv

model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3
hidden=256
run=1
epoch=2
walks=2
updateTimes=3

# pMethodList=(random) 
# AggreList=(mean)

# batch_size=(45471 22736 11368 5684 2842 1421)
# # batch_size=(45471 22736 11368 5684 2842 1421)
# fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
# hiddenList=(32 64 128 256)

# pMethodList=(random range) 
# pMethodList=(range_init_graph_partition random_init_graph_partition balanced_init_graph_partition) 

# AggreList=(mean lstm)
# AggreList=(mean )
# pMethodList=( random_init_graph_partition )

# batch_size=(45471 22736 11368 5684 2842 1421)
# layersList=(3)
# fan_out_list=(25,35,40 )
# hiddenList=(32 64 )



pMethodList=(range_init_graph_partition random_init_graph_partition balanced_init_graph_partition) 


AggreList=(lstm)
batch_size=(45471 22736 11368 5684 2842 1421)
layersList=(3)
fan_out_list=(25,35,40 25,35,80)
hiddenList=(32)
# pMethodList=(range_init_graph_partition random_init_graph_partition) 
pMethodList=(range_init_graph_partition ) 
fan_out_list=(25,35,80)
hiddenList=(64)
AggreList=(lstm)
walks=3
updateTimes=5
mkdir ${savePath}
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --num-batch $nb \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --walks $walks \
                                                --update-times $updateTimes \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done

# layersList=(4)
# fan_out_list=(25,35,40,40 )
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --walks $walks \
#                                                 --update-times $updateTimes \
#                                                 --log-indent 1 \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done

# layersList=(5)
# fan_out_list=(25,35,40,40,40 )
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --walks $walks \
#                                                 --update-times $updateTimes \
#                                                 --log-indent 1 \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done

# layersList=(6)
# fan_out_list=(25,35,40,40,40,40 )
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --walks $walks \
#                                                 --update-times $updateTimes \
#                                                 --log-indent 1 \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done
