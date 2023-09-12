import os
import ruamel.yaml
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_yaml_path', type=str, required=True, help="tokenizer count jsonl")
    parsed = parser.parse_args()
    return parsed


def add_streams(yaml_data, local_cache_dir, s3_shard_streams_path=None):
    s3_shard_list = []
    with open(s3_shard_streams_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            s3_shard_list.append(str(line.strip()))

    yaml_data['train_loader']['dataset']['streams'] = {}
    for idx, shard_path in enumerate(s3_shard_list):
        new_streams = {
            f'cache_{idx}': {
                'local': f'{local_cache_dir}/cache_{idx}',
                'remote': f'{shard_path}',
                'split': 'train',
                # 'proportion': 0.9
            },
        }
        yaml_data['train_loader']['dataset']['streams'].update(new_streams)
    return yaml_data

def get_yaml_data(yaml, yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.load(file)
    return data

def main():
    args = parse_args()

    yaml = ruamel.yaml.YAML()
    yaml_data = get_yaml_data(yaml, args.base_yaml_path)

    yaml_data = add_streams(
        yaml_data=yaml_data,
        local_cache_dir='./cache_stream',
        s3_shard_streams_path='shards_list.txt')

    OUT_YAML_PATH = 'custom.yaml'
    with open(OUT_YAML_PATH, 'w') as file:
        yaml.dump(yaml_data, file)

if __name__ == "__main__":
    main()


"""
python generate_yaml_config.py
"""

