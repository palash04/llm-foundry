import os
import ruamel.yaml
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
import ast

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_yaml_path', type=str, required=True, help="tokenizer count jsonl")
    parser.add_argument('--data_path', type=str, required=True, help="txt path containing data to be trained on")
    parser.add_argument('--data_name', type=str, required=True, help="txt path containing data to be trained on")
    parser.add_argument('--sources_list', type=str, nargs="+", help="list of sources", required=False)
    parser.add_argument("--epochs_list", type=int, nargs="+", help="List of epochs", required=False)
    parser.add_argument("--use_mosaic_repeat", action="store_true", help="Enable the mosaic repeat parameter")
    
    parsed = parser.parse_args()
    return parsed

def get_s3_paths_list(data_path, train_eval, data_name):
    s3_paths_list = []
    txt_path = os.path.join(data_path, train_eval, data_name)
    # txt_path = f'{data_path}/{train_eval}/{data_name}'
    with open(f'{txt_path}', 'r') as file:
        lines = file.readlines()
        for line in lines:
            s3_paths_list.append(str(line.strip()))
    return s3_paths_list
    

def add_streams(yaml_data, local_cache_dir, data_path, data_name, sources_list, epochs_list, use_mosaic_repeat):
    train_s3_paths_list = get_s3_paths_list(data_path, 'train', data_name)
    eval_s3_paths_list = get_s3_paths_list(data_path, 'eval' , data_name)
    

    # Train dataloader config
    if sources_list is not None:
        sources_dict = {}
        for idx in range(len(sources_list)):
            sources_dict[sources_list[idx]] = epochs_list[idx]
        if len(train_s3_paths_list) > 0:
            yaml_data['train_loader']['dataset']['streams'] = {}
            for idx, shard_path in enumerate(train_s3_paths_list):
                repeat_val = 1
                for key, value in sources_dict.items():
                    if key in shard_path:
                        repeat_val = value
                cache_name = f'cache_train_{idx}'
                if use_mosaic_repeat:
                    new_streams = {
                        f'{cache_name}': {
                            'local': f'{local_cache_dir}/{cache_name}',
                            'remote': f'{shard_path}',
                            # 'split': 'train',
                            # 'proportion': 0.9
                            'repeat': repeat_val
                        },
                    }
                    yaml_data['train_loader']['dataset']['streams'].update(new_streams)
                else:
                    for repeat_idx in range(repeat_val):
                        cache_name_ = cache_name + f'_repeat_{repeat_idx}'
                        new_streams = {
                            f'{cache_name_}': {
                                'local': f'{local_cache_dir}/{cache_name_}',
                                'remote': f'{shard_path}',
                                # 'split': 'train',
                                # 'proportion': 0.9
                            },
                        }
                        yaml_data['train_loader']['dataset']['streams'].update(new_streams)
    else:
        if len(train_s3_paths_list) > 0:
            yaml_data['train_loader']['dataset']['streams'] = {}
            for idx, shard_path in enumerate(train_s3_paths_list):
                cache_name = f'cache_train_{idx}'
                new_streams = {
                    f'{cache_name}': {
                        'local': f'{local_cache_dir}/{cache_name}',
                        'remote': f'{shard_path}',
                        # 'split': 'train',
                        # 'proportion': 0.9
                    },
                }
                yaml_data['train_loader']['dataset']['streams'].update(new_streams)
    
    # Eval dataloader config
    if len(eval_s3_paths_list) > 0:
        yaml_data['eval_loader']['dataset']['streams'] = {}
        for idx, shard_path in enumerate(eval_s3_paths_list):
            cache_name = f'cache_eval_{idx}'
            new_streams = {
                f'{cache_name}': {
                    'local': f'{local_cache_dir}/{cache_name}',
                    'remote': f'{shard_path}',
                    # 'split': 'train',
                    # 'proportion': 0.9
                },
            }
            yaml_data['eval_loader']['dataset']['streams'].update(new_streams)
    return yaml_data

def get_yaml_data(yaml, yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.load(file)
    return data

def main():
    args = parse_args()

    yaml = ruamel.yaml.YAML()
    yaml_data = get_yaml_data(yaml, args.base_yaml_path)

    data_path = args.data_path
    data_name = args.data_name
    sources_list = args.sources_list
    epochs_list = args.epochs_list
    use_mosaic_repeat = args.use_mosaic_repeat
    if sources_list is not None:
        assert len(sources_list) == len(epochs_list), "length of sources list must be equal to length of epochs list."
    
    yaml_data = add_streams(
        yaml_data=yaml_data,
        local_cache_dir='./cache_stream',
        data_path=data_path,
        data_name=data_name,
        sources_list=sources_list,
        epochs_list=epochs_list,
        use_mosaic_repeat=use_mosaic_repeat)

    OUT_YAML_PATH = 'custom.yaml'
    with open(OUT_YAML_PATH, 'w') as file:
        yaml.dump(yaml_data, file)

if __name__ == "__main__":
    main()


"""

BASE_YAML_PATH='base_mpt3b.yaml'
DATA_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data'
DATA_NAME='tok_a.txt'
SOURCES_LIST='newspaper ndl indicwiki mc4indic'
EPOCHS_LIST='1 2 3 2'
python /raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data/generate_custom_yaml_config.py --base_yaml_path $BASE_YAML_PATH --data_path $DATA_PATH --data_name $DATA_NAME --sources_list $SOURCES_LIST --epochs_list $EPOCHS_LIST 

BASE_YAML_PATH='base_mpt3b.yaml'
DATA_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data'
DATA_NAME='tok_a.txt'
SOURCES_LIST='newspaper ndl indicwiki mc4indic'
EPOCHS_LIST='1 2 3 2'
python /raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data/generate_custom_yaml_config.py --base_yaml_path $BASE_YAML_PATH --data_path $DATA_PATH --data_name $DATA_NAME --use_mosaic_repeat --sources_list $SOURCES_LIST --epochs_list $EPOCHS_LIST 

BASE_YAML_PATH='base_mpt1b.yaml'
DATA_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data'
DATA_NAME='tok_a.txt'
SOURCES_LIST=''
EPOCHS_LIST=''
python /raid/palash.kamble/LLM/llm-foundry/scripts/data_prep/custom_data/generate_custom_yaml_config.py --base_yaml_path $BASE_YAML_PATH --data_path $DATA_PATH --data_name $DATA_NAME

python train/custom/generate_yaml_config.py --base_yaml_path $BASE_YAML_PATH --data_path $DATA_PATH --data_name $DATA_NAME

"""
