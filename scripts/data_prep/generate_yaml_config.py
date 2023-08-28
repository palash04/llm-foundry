import os
import ruamel.yaml
from argparse import ArgumentParser
import boto3
from config import *
from tqdm import tqdm

s3_client = boto3.client('s3')
bucket_name = 'llm-spark'

def add_streams(yaml_data, local_cache_dir, s3_mds_base_path, start_shard_idx, end_shard_idx, lang):
    if lang == 'eng':
        lang = '__label__en'
    yaml_data['train_loader']['dataset']['streams'] = {}
    for shard_idx in tqdm(range(start_shard_idx, end_shard_idx+1), desc='Iterating shards'):
        index_path = os.path.join(s3_mds_base_path, f'sh_{shard_idx}', f'{lang}', 'train', 'index.json')
        try:
            _ = s3_client.get_object(Bucket=bucket_name, Key=index_path)
            shard_path = os.path.join('s3://llm-spark', s3_mds_base_path, f'sh_{shard_idx}', f'{lang}')
            new_streams = {
                f'cache_{shard_idx}_{lang}': {
                    'local': f'{local_cache_dir}/cache_{shard_idx}_{lang}',
                    'remote': f'{shard_path}',
                    'split': 'train',
                    # 'proportion': 0.9
                },
            }
            yaml_data['train_loader']['dataset']['streams'].update(new_streams)
            # yaml_data['train_loader']['dataset']['streams']=new_streams
        except:
            continue
    return yaml_data

def get_yaml_data(yaml, yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.load(file)
    return data

def main():
    yaml = ruamel.yaml.YAML()
    yaml_data = get_yaml_data(yaml, BASE_YAML_PATH)
    
    yaml_data['run_name']=RUN_NAME
    yaml_data['save_folder']=CHECKPOINT_PATH
    yaml_data['save_interval']=SAVE_INTERVAL
    yaml_data['global_train_batch_size']=GLOBAL_TRAIN_BATCH_SIZE
    yaml_data['device_train_microbatch_size']=DEVICE_TRAIN_MICROBATCH_SIZE
    yaml_data['train_loader']['dataset']['cache_limit']=CACHE_LIMIT
    yaml_data['max_seq_len']=MAX_SEQ_LEN
    yaml_data['loggers']['wandb']['init_kwargs']['mode']=WANDB_MODE
    yaml_data['max_duration']=MAX_DURATION

    yaml_data = add_streams(
        yaml_data=yaml_data,
        local_cache_dir=LOCAL_CACHE_DIR,
        s3_mds_base_path=S3_MDS_BASE_PATH, 
        start_shard_idx=START_SHARD_IDX, 
        end_shard_idx=END_SHARD_IDX, 
        lang=LANG)

    with open(OUT_YAML_PATH, 'w') as file:
        yaml.dump(yaml_data, file)
    print(OUT_YAML_PATH)

if __name__ == "__main__":
    main()


"""
python generate_yaml_config.py
"""

