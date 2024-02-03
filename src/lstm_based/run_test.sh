#!/bin/bash

source ~/dev/pyvenv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

RUN_DIR=$PWD

RUN_TRACE_DIR="./run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_TRACE_DIR}/test_results_$today.out"
echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

nohup python3 -u test_main.py \
    --dataset_root . \
    --modeldata_root . \
    --vocab_dir ${RUN_DIR}/../data.100d \
    --vocab_dim 100 \
    --eval true >>${RUN_LOG} 2>&1 &
