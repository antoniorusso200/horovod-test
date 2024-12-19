#!/bin/bash
set -xe

export DATA=/data/scratch/cifar-10
NUM_EPOCH=20
pip install -r requirements.txt
colossalai run --nproc_per_node 2 --host 143.225.229.155,143.225.229.156 --master_addr 143.225.229.156 train.py --epoch ${NUM_EPOCH} --interval 10 #--target_acc 0.84 
colossalai run --nproc_per_node 2 eval.py --epoch ${NUM_EPOCH} --checkpoint ./checkpoint


# TODO: skip ci test due to time limits, train.py needs to be rewritten.

# for plugin in "torch_ddp" "torch_ddp_fp16" "low_level_zero"; do
#     
# done
