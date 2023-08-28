# Local DGX
# BASE_YAML_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/train/yamls/pretrain/base-mpt-1b-multi-stream.yaml'
# OUT_YAML_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/train/yamls/pretrain/mpt-1b-multi-stream-eng.yaml'
# LOCAL_CACHE_DIR='/raid/palash.kamble/LLM/tmp/cache_stream'
# S3_MDS_BASE_PATH='llm/pretrain_data/mds_data/snap_2023_23/'
# START_SHARD_IDX=0
# END_SHARD_IDX=2 # inclusive
# LANG='eng' # eng | mix
# RUN_NAME='mpt1B_eng_20BT_dgx_v1'
# CHECKPOINT_PATH=f'/raid/palash.kamble/LLM/checkpoints/{RUN_NAME}'
# SAVE_INTERVAL='2ba'
# GLOBAL_TRAIN_BATCH_SIZE=1024
# DEVICE_TRAIN_MICROBATCH_SIZE=32
# CACHE_LIMIT='20gb'
# MAX_SEQ_LEN=2048
# WANDB_MODE='online' # online | offline
# MAX_DURATION='5ba' # 1ep | 1ba | 10ba | 43ba | anything you wish


# MCLI
BASE_YAML_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/train/yamls/pretrain/base-mpt-1b-multi-stream.yaml'
OUT_YAML_PATH='/raid/palash.kamble/LLM/llm-foundry/scripts/train/yamls/pretrain/mcli-mpt-1b-multi-stream-eng.yaml'
LOCAL_CACHE_DIR='./cache_stream'
S3_MDS_BASE_PATH='llm/pretrain_data/mds_data/snap_2023_23/'
START_SHARD_IDX=0
END_SHARD_IDX=30 # inclusive
LANG='eng' # eng | mix
RUN_NAME='mcli-mpt1B_eng_20BT_v2'
CHECKPOINT_PATH=f'/raid/palash.kamble/LLM/checkpoints/{RUN_NAME}'
SAVE_INTERVAL='100ba'
GLOBAL_TRAIN_BATCH_SIZE=1024
DEVICE_TRAIN_MICROBATCH_SIZE=32
CACHE_LIMIT='900gb'
MAX_SEQ_LEN=2048
WANDB_MODE='online' # online | offline
MAX_DURATION='1ep' # 1ep | 1ba | 10ba | 43ba | anything you wish
