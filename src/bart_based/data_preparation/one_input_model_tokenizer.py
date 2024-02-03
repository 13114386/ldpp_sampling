from __future__ import unicode_literals, print_function, division
import os, json
from file_utils import ChunkSaver
from model_tokenizer import ModelTokenizer

class OneInputFileModelTokenizer(ModelTokenizer):
    def process(
        self,
        args,
        split_type,
        tokenizer,
        model_type,
        logger
    ):
        dataset_root = os.path.join(args.dataset_root, args.datasource_name)
        source_dir = os.path.join(dataset_root, args.source_folder)
        source_path = os.path.join(source_dir, f'{split_type}{args.source_file_ext}')
        output_dir = os.path.join(dataset_root, args.output_folder, model_type)
        os.makedirs(output_dir, exist_ok=True)
        output0_path = os.path.join(output_dir, f'{split_type}.{args.column_names[0]}{args.output_file_ext}')
        output1_path = os.path.join(output_dir, f'{split_type}.{args.column_names[1]}{args.output_file_ext}')

        error_lines = []
        composed = {args.column_names[0]: [],
                    args.column_names[1]: []}
        save_ids = {args.column_names[0]: output0_path,
                    args.column_names[1]: output1_path}
        with ChunkSaver(save_ids, args.chunk_size, convert_json=False) as saver, \
            open(source_path, "r", encoding="utf-8") as src_fp:
            for index, line in enumerate(src_fp):
                try:
                    data = json.loads(line)
                    col0 = tokenizer(data[args.column_names[0]],
                                     truncation=args.truncation,
                                     max_length=args.max_length)
                    col1 = tokenizer(data[args.column_names[1]],
                                     truncation=args.truncation,
                                     max_length=args.max_length)
                except Exception as ex:
                    error_lines.append(index)
                    logger.error(str(ex))
                    continue

                try:
                    col0_data_encoded = json.dumps(col0.data, ensure_ascii=False)
                    col1_data_encoded = json.dumps(col1.data, ensure_ascii=False)
                    composed[args.column_names[0]].append(col0_data_encoded)
                    composed[args.column_names[1]].append(col1_data_encoded)
                except (json.decoder.JSONDecodeError, UnicodeEncodeError) as ex:
                    error_lines.append(index)
                    logger.error(str(ex))
                    continue

                if saver(features=composed, index=index, last_save=False):
                    [composed[k].clear() for k, v in composed.items()]
            # If any remaining
            if saver(features=composed, index=index, last_save=True):
                [composed[k].clear() for k, v in composed.items()]

        if len(error_lines):
            logger.warning(f"{args.split_type} has {len(error_lines)} improperly parsed articles of indexed list {error_lines}")
