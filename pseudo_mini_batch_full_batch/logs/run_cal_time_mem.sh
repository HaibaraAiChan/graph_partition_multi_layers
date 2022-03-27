#!/bin/bash

model=sage
file=arxiv
# aggre=lstm
# pMethod=range

aggreList=(mean lstm)
pMethodList=(random range)
eval=False
for aggre in ${aggreList[@]}
do
        for pMethod in ${pMethodList[@]}
        do
            resPath=${model}/res/${aggre}/$pMethod
            mkdir $resPath

            python calculate_time_mem.py \
            --aggre $aggre \
            --selection-method $pMethod \
            --save-path ${resPath}/${file}_${model}_eval_${eval}_pseudo_ \
            > ${resPath}/${file}_${model}_eval_${eval}_mem.log

            python calculate_compute_efficiency.py \
            --aggre $aggre \
            --selection-method $pMethod \
            --save-path ${resPath}/${file}_${model}_eval_${eval}_pseudo_ \
            > ${resPath}/${file}_${model}_eval_${eval}_eff.log
        done
done
# model=sage
# file=arxiv
# aggre=mean
# eval=False

# python calculate_time_mem.py \
# --save-path ${model}/res/${aggre}/${file}_${model}_eval_${eval}_pseudo_ \
# > ${model}/res/${aggre}/${file}_${model}_eval_${eval}_mem.log

# python calculate_compute_efficiency.py \
# --save-path ${model}/res/${aggre}/${file}_${model}_eval_${eval}_pseudo_ \
# > ${model}/res/${aggre}/${file}_${model}_eval_${eval}_eff.log
