from argparse import ArgumentParser, Namespace
import json
from collections import defaultdict
import os

DASHLINE = '-'.join(['' for _ in range(100)])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, required=True, help="tokenizer count jsonl")
    parser.add_argument('--s3_mds_base_path', type=str, required=True, help="s3 mds data base path")
    parser.add_argument('--lang_list', nargs='+', type=str, help='list of languages')
    parser.add_argument('--tokens_in_billions_list', nargs='+', type=int, help='tokens list in billions corresponding to each language')
    parser.add_argument('--epochs_list', nargs='+', type=int, help='number of epochs list corresponding to each language')
    parsed = parser.parse_args()
    return parsed


def sort_by_token_count(tuples_list):
    return sorted(tuples_list, key=lambda x: x[1], reverse=True)


def get_list(lang_list, tokens_in_billions_list, epochs_list, jsonl_path, s3_mds_base_path):
    all_data_set = set()
    with open(jsonl_path, "r") as input_file:
        for line in input_file:
            try:
                data = json.loads(line)
                all_data_set.add(tuple(data.items()))
            except:
                # print(line)
                pass
    all_data = list(all_data_set)
    
    lang_dict = defaultdict(list)

    for data in all_data:
        # (('snap', 'exp2'), ('shard_id', 101), ('lang', '__label__te'), ('token_count', 2068480))
        shard_id = data[1][1]
        lang = data[2][1]
        token_count = data[3][1]

        lang_dict[lang].append((shard_id, token_count))
    
    s3_sharded_list = []
    for idx in range(len(lang_list)):
        lang_shards = []
        lang = lang_list[idx]
        lang = f'__label__{lang}'
        lang_tokens = tokens_in_billions_list[idx] * 1e9
        epochs = epochs_list[idx]
        total_lang_collected = 0
        if lang in lang_dict:
            list_of_token_counts = lang_dict[lang]
            while lang_tokens > 0 and len(list_of_token_counts) > 0:
                list_of_token_counts = sorted(list_of_token_counts, key=lambda x: x[1], reverse=True)
                for ep in range(epochs):
                    lang_shards.append((lang, list_of_token_counts[0][0], list_of_token_counts[0][1]))
                total_lang_collected += list_of_token_counts[0][1]
                lang_tokens -= list_of_token_counts[0][1]
                list_of_token_counts = list_of_token_counts[1:]

        print(f'Total tokens for {lang} collected (in Billions): ', total_lang_collected/1e9)

        for lang_shard in lang_shards:
            lang = lang_shard[0]
            shard_id = lang_shard[1]
            s3_full_path = os.path.join(s3_mds_base_path, f'sh_{shard_id}', lang)
            s3_sharded_list.append(s3_full_path)

    return s3_sharded_list
    
def main():
    args = parse_args()
    jsonl_path = args.jsonl_path
    s3_mds_base_path = args.s3_mds_base_path
    lang_list = args.lang_list
    tokens_in_billions_list = args.tokens_in_billions_list
    epochs_list = args.epochs_list

    s3_sharded_list = get_list(
        lang_list=lang_list, 
        tokens_in_billions_list=tokens_in_billions_list, 
        epochs_list=epochs_list,
        jsonl_path = jsonl_path,
        s3_mds_base_path=s3_mds_base_path, )

    print('Number of streams: ', len(s3_sharded_list))

    file_name = "shards_list.txt"
    # Write the list to the text file
    with open(file_name, 'w') as file:
        for item in s3_sharded_list:
            file.write(str(item) + '\n')


if __name__ == "__main__":
    main()

"""
python create_shard_paths_exp_multi_lang.py --jsonl_path /raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/exp3.jsonl --s3_mds_base_path s3://llm-spark/llm/pretrain_data/mds_data/exp3_new/ --lang_list hi ta --tokens_in_billions_list 3 1 --epochs_list 1 1
"""
