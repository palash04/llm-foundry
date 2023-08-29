# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import numpy as np
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional
import gzip
import pickle
import boto3
from collections import defaultdict
import datasets as hf_datasets
from datasets import Dataset
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
import concurrent.futures
import json


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

s3_client = boto3.client('s3')
bucket_name = 'llm-spark'
DASHLINE = '-'.join(['' for _ in range(100)])

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--s3_base_path', type=str, required=True)
    parser.add_argument('--shard_start_idx', type=int, required=True)
    parser.add_argument('--shard_end_idx', type=int, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')
    # parser.add_argument('--split', type=str, default='train')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--max_workers', type=int, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False, default='')

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def build_hf_dataset(
    hf_dataset,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
    tokenizer_name: str = '',
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """

    # hf_dataset = Dataset.load_from_disk(arrow_path)
    

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase')
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(hf_dataset=hf_dataset,
                                      tokenizer=tokenizer,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      no_wrap=no_wrap,
                                      tokenizer_name=tokenizer_name)
    return dataset


def _est_progress_denominator(total_samples: int, chars_per_sample: int,
                              chars_per_token: int, mode: ConcatMode,
                              max_length: int):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}

languages = {
    'non_english': ['__label__as', '__label__pa', '__label__bn', '__label__or', '__label__gu', '__label__mr', '__label__kn', '__label__te', '__label__ml', '__label__ta', '__label__hi'],
    'english': ['__label__en']
}

def main_helper(args: Namespace, shard_idx: int):
    
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'bytes'}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str'}

    # Get samples
    s3_base_path = args.s3_base_path
    shard = f'sh_{shard_idx}'
    lang = args.lang

    if lang == 'eng':
        lang_list = languages['english']
    else:
        lang_list = languages['non_english']
    
    lang_stats = []
    for lang in lang_list:
        shard_path = os.path.join(s3_base_path, shard, lang)
        print(f'Working for shard {shard_idx} lang {lang}...')
        is_data = True
        try:
            hf_dataset = Dataset.load_from_disk(shard_path)
        except:
            is_data = False
        if not is_data:
            lang_stats.append((shard_idx, lang, 0))
        else:
            dataset = build_hf_dataset(hf_dataset=hf_dataset,
                                mode=mode,
                                max_length=args.concat_tokens,
                                bos_text=args.bos_text,
                                eos_text=args.eos_text,
                                no_wrap=args.no_wrap,
                                tokenizer=tokenizer,
                                tokenizer_name=args.tokenizer_name)

            token_count: int = 0
            if 's3://llm-spark' in args.out_root:
                path = os.path.join(args.out_root, f'{shard}', f'{lang}', 'train')
            else:
                path = os.path.join(args.out_root)

            with MDSWriter(columns=columns,
                        out=path,
                        compression=args.compression,
                        max_workers=32) as out:
                for sample in dataset:
                    tokens = np.frombuffer(sample['tokens'], dtype=np.int64).copy()
                    token_count += len(tokens)
                    out.write(sample)
            lang_stats.append((shard_idx, lang, token_count))
    return lang_stats

def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    shard_idx_list = list(range(args.shard_start_idx, args.shard_end_idx+1))
    # results = [main_helper(args, 0)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(main_helper, args, shard_idx) for shard_idx in shard_idx_list]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    logs_list = []
    for res in results:
        for lang_token in res:
            shard_id = lang_token[0]
            lang = lang_token[1]
            token_count = lang_token[2]
            logs_list.append({'snap':'snap_2023_14', 'shard_id': shard_id, 'lang': lang, 'token_count': token_count})
            # shard_dict[shard_id] += token_count
            # lang_token_dict[lang] += token_count

    output_file_path = f'{args.tokenizer_name}_logs.jsonl'
    with open(output_file_path, "a") as output_file:
        for log_dict in logs_list:
            json.dump(log_dict, output_file)  # Write the dictionary to the file
            output_file.write("\n")  # Add a newline to separate records

if __name__ == '__main__':
    main(parse_args())

# MPT
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/mpt/snap_2023_14/ --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 130 --lang non_eng --max_workers 192 --tokenizer_name mpt
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/mpt/snap_2023_14/ --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 130 --lang eng --max_workers 192 --tokenizer_name mpt

# LLAMA
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/llama/snap_2023_14/ --concat_tokens 2048 --tokenizer meta-llama/Llama-2-7b-hf --eos_text '</s>' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 130 --lang non_eng --max_workers 192 --tokenizer_name llama
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/llama/snap_2023_14/ --concat_tokens 2048 --tokenizer meta-llama/Llama-2-7b-hf --eos_text '</s>' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 99 --lang eng --max_workers 192 --tokenizer_name llama

# ai4bharat/indic-bert
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/indic_bert/snap_2023_14/ --concat_tokens 2048 --tokenizer ai4bharat/indic-bert --eos_text '[SEP]' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 5 --lang non_eng --max_workers 30 --tokenizer_name indicbert
# time python convert_dataset_arrow_parallelized.py --out_root s3://llm-spark/llm/pretrain_data/mds_data/indic_bert/snap_2023_14/ --concat_tokens 2048 --tokenizer ai4bharat/indic-bert --eos_text '[SEP]' --compression zstd --s3_base_path s3://llm-spark/llm/pretrain_data/arrow_data/snap_2023_23 --shard_start_idx 0 --shard_end_idx 25 --lang eng --max_workers 25 --tokenizer_name indicbert
