## Python virtual environment
    Refer to pyvenv_requirements.txt.


## Process to build datasets
- Download CNNDM datasets into separate folders such as cnndm/.<br>
  (You may use Huggingface datasets APIs to get them).
- CNNDM contain:
    - train.json (287113 lines)
    - validation.json (13368 lines)
    - test.json (11490 lines)


### Note: all following running script examples should be one-line commands without backslash. Multiple lined examples here are for readability.

### For BART based model: under src/bart_based
#### Build CNNDM dataset: from src/bart_based/data_preparation
- An example for pre-preparing train dataset:
    ```
    . model_token_dataset_builder.sh \
        --dataset_root /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
        --datasource_name cnndm \
        --source_folder directory/to/downloaded/cnndm \
        --output_folder folder/to/output_files \
        --source_file_ext .json \
        --output_file_ext .json \
        --split_type train
        --column_names [article,highlights] \
        --tokenizer_name facebook/bart-base \
        --pairing_names none
    ```
Note: to pre-prepare validation or test, replace train split_type with validation or test.

#### Runtime:
- An example for training:
    ```
    . run.training.mgpu.sh \
        --dataset_root  /root/directory/to/runtime/dataset \
        --dataset_folder  data_folder_under_dataset_root \
        --datasource cnndm \
        --base_model_pretrained_name facebook/bart-base \
        --tokenizer_name facebook/bart-base \
        --split_type "[\"train\",\"validation\"]" \
        --pair_type "[\"article\",\"highlights\"]" \
        --early_stop_count_on_rouge 3 \
        --pyvenv_dir  /your/python/venv/bin/directory \
        --gpu_ids  0,1 \
        --process_port_id 9999
    ```
- An example for test:
    ```
    . run.eval.mgpu.sh \
        --dataset_root  /root/directory/to/runtime/dataset \
        --dataset_folder  data_folder_under_dataset_root \
        --datasource cnndm \
        --base_model_pretrained_name facebook/bart-base \
        --tokenizer_name facebook/bart-base \
        --split_type "[\"test\"]" \
        --pair_type "[\"article\",\"highlights\"]" \
        --pyvenv_dir  /your/python/venv/bin/directory \
        --gpu_ids  0,1 \
        --process_port_id 9999

    ```
Note: For single GPU, specify gpu_ids to 0.
