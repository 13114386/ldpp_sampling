#!/bin/bash


help()
{
    echo "Usage: model_token_dataset_builder.sh --dataset_root   [dataset_root_directory]"
    echo "                                    --datasource_name  [cnndm | gigaword]"
    echo "                                    --source_folder    [cnndm:downloaded | gigaword:downloaded/org_data]"
    echo "                                    --output_folder    [model_tokenization]"
    echo "                                    --source_file_ext  [cnndm.json | gigaword:.txt]"
    echo "                                    --output_file_ext  [.json]"
    echo "                                    --split_type       cnndm:[train|validation|test] | gigaword:[train|dev|test]"
    echo "                                    --column_names     cnndm:[article,highlights] | gigaword:[document,summary]"
    echo "                                    --tokenizer_name   [base:facebook/bart-base | cnndm:facebook/bart-large-cnn | gigaword:a1noack/bart-large-gigaword]"
    echo "                                    --pairing_names    [cnndm:none | gigaword:[src,tgt]]"
}


#LONG=source_folder:,output_folder:,split_types:,build_vocab:,use_corefs:,analyse_corefs:,build_compose:,build_stype2id:
#OPTS=$(getopt --longoptions $LONG -- "$@")

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=20
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

#eval set -- "$OPTS"

while :
do
  case "$1" in
    --dataset_root )
      dataset_root="$2"
      shift 2
      ;;
    --datasource_name )
      datasource_name="$2"
      shift 2
      ;;
    --source_folder )
      source_folder="$2"
      shift 2
      ;;
    --output_folder )
      output_folder="$2"
      shift 2
      ;;
    --source_file_ext )
      source_file_ext="$2"
      shift 2
      ;;
    --output_file_ext )
      output_file_ext="$2"
      shift 2
      ;;
    --split_type )
      split_type="$2"
      shift 2
      ;;
    --column_names )
      column_names="$2"
      shift 2
      ;;
    --tokenizer_name )
      tokenizer_name="$2"
      shift 2
      ;;
    --pairing_names )
      pairing_names="$2"
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


source ~/dev/pyvenv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/..:$PWD/../.."

RUN_TRACE_DIR="${dataset_root}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

WORKING_FOLDER_NAME="`basename $PWD`"

today=`date '+%Y_%m_%d_%H_%M'`;
LOG_FILE="${RUN_TRACE_DIR}/${WORKING_FOLDER_NAME}_build_${split_type}_$today.log"
echo "Log file path:"
echo "  $LOG_FILE"
echo $HOSTNAME >${LOG_FILE}

build_args="--dataset_root ${dataset_root} --datasource_name ${datasource_name}"
build_args="${build_args} --source_folder ${source_folder} --output_folder ${output_folder}"
build_args="${build_args} --source_file_ext ${source_file_ext} --output_file_ext ${output_file_ext}"
build_args="${build_args} --split_type ${split_type} --column_names ${column_names}"
build_args="${build_args} --tokenizer_name ${tokenizer_name}"

if [ "${pairing_names}" != "none" ]; then
    build_args="${build_args} --pairing_names ${pairing_names}"
fi

echo "Build Args:"
echo "  ${build_args}"

nohup python3 -u main.py ${build_args} >>${LOG_FILE} 2>&1 &
