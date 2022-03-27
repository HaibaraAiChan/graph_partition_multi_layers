#!/bin/bash

File=pseudo_mini_range.py

# Data=reddit
# Aggre=mean
# batch_size=( 9590 4795 2400 1200)
# fan_out=10,25
# layers=2
# # Data=cora
Data=karate
Aggre=mean
batch_size=(2 3 4 5 6 7 8 14 16 24)
# # # batch_size=(3 3 3 3 3 3)
# # fan_out=2
# # # layers=1
# fan_out=2,2
# epoch=200
# dropout=0.5
# layers=2
fan_out=2,2,2
layers=3
epoch=6
dropout=0.0
# # mkdir logs
# for bs in ${batch_size[@]}
# do
#     python $File \
#         --dataset $Data \
#         --aggre $Aggre \
#         --selection-method range \
#         --batch-size $bs \
#         --num-epochs 6 \
#         --num-layers $layers \
#         --fan-out $fan_out \
#         --eval-every 5 
# done
for bs in ${batch_size[@]}
do
    python $File \
        --dataset $Data \
        --aggre $Aggre \
        --selection-method range \
        --batch-size $bs \
        --num-epochs $epoch \
        --num-layers $layers \
        --dropout $dropout \
        --fan-out $fan_out \
        --eval-every 5 &> logs/${layers}_layer_${Data}_${Aggre}_pseudo_mini_range_bs_${bs}_fan-out_${fan_out}.log
done

# for bs in ${batch_size[@]}
# do
#     python $File \
#         --dataset $Data \
#         --aggre $Aggre \
#         --selection-method range \
#         --batch-size $bs \
#         --num-epochs 6 \
#         --num-layers $layers \
#         --fan-out $fan_out \
#         --dropout 0.0 \
#         --eval-every 5 &> logs/${layers}_layer_${Data}_${Aggre}_pseudo_mini_range_reconstruct_bs_${bs}_fan-out_${fan_out}.log
# done
# Data=cora
# Aggre=mean
# batch_size=(10 20 40 60 70 140)
# fan_out=10,25
# layers=2
# epoch=300
# dropout=0.5
# # mkdir logs
# for bs in ${batch_size[@]}
# do
#     python $File \
#         --dataset $Data \
#         --aggre $Aggre \
#         --selection-method range \
#         --batch-size $bs \
#         --num-epochs $epoch \
#         --num-layers $layers \
#         --fan-out $fan_out \
#         --dropout $dropout \
#         --eval-every 5 &> logs/${layers}_layer_${Data}_${Aggre}_pseudo_mini_range_bs_${bs}_fan-out_${fan_out}_${epoch}_epochs.log
# done

# Data=karate
# Aggre=mean
# batch_size=(2 3 4 6 7 8 14)
# fan_out=1,1
# layers=2

# for bs in ${batch_size[@]}
# do
#     python $File \
#         --dataset $Data \
#         --aggre $Aggre \
#         --selection-method range \
#         --batch-size $bs \
#         --num-epochs 6 \
#         --num-layers $layers \
#         --fan-out $fan_out \
#         --eval-every 5 &> logs/${layers}_layer_${Data}_${Aggre}_pseudo_mini_range_bs_${bs}_fan-out_${fan_out}.log
# done

# fan_out=2,2,2
# layers=3
# for bs in ${batch_size[@]}
# do
#     python $File \
#         --dataset $Data \
#         --aggre $Aggre \
#         --selection-method range \
#         --batch-size $bs \
#         --num-epochs 6 \
#         --num-layers $layers \
#         --fan-out $fan_out \
#         --eval-every 5 &> logs/${layers}_layer_${Data}_${Aggre}_pseudo_mini_range_bs_${bs}_fan-out_${fan_out}.log
# done