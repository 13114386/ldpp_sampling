#!/bin/bash


help()
{
    echo "Usage: run.phase.mgpu.sh "
    echo "      --dataset_root                [dataset root directory]"
    echo "      --dataset_folder              [""|the data folder relevant to the dataset_root]"
    echo "      --datasource                  [cnndm|gigaword]"
    # echo "      --config_folder               [cnndm|gigaword]"
    echo "      --base_model_pretrained_name  [facebook/bart-base|facebook/bart-large-cnn|a1noack/bart-large-gigaword]"
    echo "      --tokenizer_name              [facebook/bart-base|facebook/bart-large-cnn|a1noack/bart-large-gigaword]"
    echo "      --split_type                  [\"train\",\"dev\"]|[\"train\",\"validation\"]"
    echo "      --pair_type                   [\"document\",\"summary\"]|[\"article\",\"highlights\"]"
    echo "      --early_stop_count_on_rouge   [3|4|-1]"
    echo "      --pyvenv_dir                  [a_python_venv_directory]"
    echo "      --gpu_ids                     [0,1,2]"
    echo "      --process_port_id             [9999]"
}

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=22
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

while :
do
  case "$1" in
    --dataset_root )
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset_folder )
      DATASET_FOLDER="$2"
      shift 2
      ;;
    --datasource )
      DATASOURCE="$2"
      shift 2
      ;;
    --base_model_pretrained_name )
      BASE_MODEL_PRETRAINED_NAME="$2"
      shift 2
      ;;
    --tokenizer_name )
      TOKENIZER_NAME="$2"
      shift 2
      ;;
    --split_type )
      DATA_SPLIT_TYPE="$2"
      shift 2
      ;;
    --pair_type )
      DATA_PAIR_TYPE="$2"
      shift 2
      ;;
    --early_stop_count_on_rouge )
      EARLY_STOP_COUNT_ON_ROUGE="$2"
      shift 2
      ;;
    --pyvenv_dir )
      PYVENV_DIR="$2"
      shift 2
      ;;
    --gpu_ids )
      GPU_IDS="$2"
      shift 2
      ;;
    --process_port_id )
      PROCESS_PORT_ID="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      # echo "Unexpected option: $1"
      # help
      break
      ;;
  esac
done


source ${PYVENV_DIR}/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

FOLDER_NAME="`basename $PWD`"

RUN_TRACE_DIR="${FOLDER_NAME}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_TRACE_DIR}/${DATASOURCE}_train_log_$today.out"

echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

echo "--modeldata_root:             ${FOLDER_NAME}"
echo "--dataset_root:               ${DATASET_ROOT}"
echo "--config_folder:              ${DATASOURCE}"
echo "--dataset_folder:             ${DATASET_FOLDER}"
echo "--base_model_pretrained_name: ${BASE_MODEL_PRETRAINED_NAME}"
echo "--tokenizer_name:             ${TOKENIZER_NAME}"
echo "--split_type:                 ${DATA_SPLIT_TYPE}"
echo "--pair_type:                  ${DATA_PAIR_TYPE}"
echo "--dataset_file:               {split_type}.{pair_type}.dataset.json"
echo "--early_stop_count_on_rouge   ${EARLY_STOP_COUNT_ON_ROUGE}"

#if [ 1 -eq 0 ]; then
PARAMS="--modeldata_root ${FOLDER_NAME} "
PARAMS=${PARAMS}"--dataset_root ${DATASET_ROOT} "
PARAMS=${PARAMS}"--dataset_folder ${DATASET_FOLDER} "
PARAMS=${PARAMS}"--config_folder ${DATASOURCE} "
PARAMS=${PARAMS}"--base_model_pretrained_name  ${BASE_MODEL_PRETRAINED_NAME} "
PARAMS=${PARAMS}"--tokenizer_name ${TOKENIZER_NAME} "
PARAMS=${PARAMS}"--use_slow_tokenizer "
PARAMS=${PARAMS}"--split_type ${DATA_SPLIT_TYPE} "
PARAMS=${PARAMS}"--pair_type ${DATA_PAIR_TYPE} "
PARAMS=${PARAMS}"--dataset_file {split_type}.{pair_type}.dataset.json "
PARAMS=${PARAMS}"--seed 19786403 "
PARAMS=${PARAMS}"--early_stop_count_on_rouge ${EARLY_STOP_COUNT_ON_ROUGE} "

MAIN_APP=train_main.py


GPU_ID_LIST=(${GPU_IDS//,/ })
NUM_GPUS=${#GPU_ID_LIST[@]}
echo "num_gpus:    ${NUM_GPUS}"

if [ "${NUM_GPUS}" -gt "1" ]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.${DATASOURCE}.yaml ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.${DATASOURCE}.yaml ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
else
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
fi

#fi
