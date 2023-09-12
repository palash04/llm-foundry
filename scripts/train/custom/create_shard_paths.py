from argparse import ArgumentParser, Namespace
import json
from collections import defaultdict
import os

DASHLINE = '-'.join(['' for _ in range(100)])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, required=True, help="tokenizer count jsonl")
    parser.add_argument('--s3_mds_base_path', type=str, required=True, help="s3 mds data base path")
    parser.add_argument('--snap', type=str, required=True, help="snap ex. snap_2023_23 ")
    parser.add_argument('--indic_ratio', type=float, required=True, help="ratio of indic data for training")
    parser.add_argument('--tokens_in_billions', type=int, required=True, help="total number of tokens for training")
    parsed = parser.parse_args()
    if parsed.indic_ratio > 1:
        raise 'ratio must be less than equal to 1'
    return parsed


def sort_by_token_count(tuples_list):
    return sorted(tuples_list, key=lambda x: x[1], reverse=True)


def get_list(english_tokens, indic_tokens, jsonl_path, s3_mds_base_path, snap):
    all_data_set = set()
    with open(jsonl_path, "r") as input_file:
        for line in input_file:
            try:
                data = json.loads(line)
                if snap == data['snap']:
                    all_data_set.add(tuple(data.items()))
            except:
                # print(line)
                pass
    all_data = list(all_data_set)
    
    lang_dict = defaultdict(list)

    for data in all_data:
        # (('snap', 'snap_2023_14'), ('shard_id', 101), ('lang', '__label__te'), ('token_count', 2068480))
        shard_id = data[1][1]
        lang = data[2][1]
        token_count = data[3][1]

        lang_dict[lang].append((shard_id, token_count))
    
    indic_shards = []
    while indic_tokens > 0:
        any_data_left = False
        for key, list_of_token_counts in lang_dict.items():
            if key != "__label__en":
                list_of_token_counts = sorted(list_of_token_counts, key=lambda x: x[1], reverse=True)
                if len(list_of_token_counts) > 0 and list_of_token_counts[0][1] > 0:
                    indic_shards.append((key, list_of_token_counts[0][0], list_of_token_counts[0][1]))
                    any_data_left = True
                    indic_tokens -= list_of_token_counts[0][1]
                    list_of_token_counts = list_of_token_counts[1:]
                    lang_dict[key] = list_of_token_counts
        if not any_data_left:
            print('No data left...exiting')
            break

    english_shards = []
    list_of_token_counts_eng = lang_dict['__label__en']
    list_of_token_counts_eng = sorted(list_of_token_counts_eng, key=lambda x: x[1], reverse=True)
    
    for shard_id_token_count in list_of_token_counts_eng:
        shard_id = shard_id_token_count[0]
        token_count = shard_id_token_count[1]
        if token_count == 0:
            break
        english_tokens -= token_count
        english_shards.append(('__label__en', shard_id, token_count))
        if english_tokens <= 0:
            break

    total_indic_collected = 0
    total_english_collected = 0
    for indic_shard in indic_shards:
        total_indic_collected += indic_shard[2]
    for english_shard in english_shards:
        total_english_collected += english_shard[2]
    
    print('Total indic tokens collected (in Billions): ', total_indic_collected/1e9)
    print('Total english tokens collected (in Billions): ', total_english_collected/1e9)

    s3_sharded_list = []

    for indic_shard in indic_shards:
        lang = indic_shard[0]
        shard_id = indic_shard[1]
        s3_full_path = os.path.join(s3_mds_base_path, snap, f'sh_{shard_id}', lang)
        s3_sharded_list.append(s3_full_path)

    for english_shard in english_shards:
        lang = english_shard[0]
        shard_id = english_shard[1]
        s3_full_path = os.path.join(s3_mds_base_path, snap, f'sh_{shard_id}', lang)
        s3_sharded_list.append(s3_full_path)

    return s3_sharded_list
    
def main():
    args = parse_args()
    jsonl_path = args.jsonl_path
    s3_mds_base_path = args.s3_mds_base_path
    snap = args.snap
    total_tokens = args.tokens_in_billions
    indic_ratio = args.indic_ratio

    english_tokens = int(total_tokens * (1 - indic_ratio))
    indic_tokens = int(total_tokens - english_tokens)

    english_tokens = int(english_tokens * 1e9)
    indic_tokens = int(indic_tokens * 1e9)

    s3_sharded_list = get_list(
        english_tokens=english_tokens, 
        indic_tokens=indic_tokens, 
        jsonl_path = jsonl_path,
        s3_mds_base_path=s3_mds_base_path, 
        snap=snap)

    print('Number of streams: ', len(s3_sharded_list))

    file_name = "shards_list.txt"
    # Write the list to the text file
    with open(file_name, 'w') as file:
        for item in s3_sharded_list:
            file.write(str(item) + '\n')


if __name__ == "__main__":
    main()

"""
python create_shard_paths.py --indic_ratio 0.2 --tokens_in_billions 20
"""
