import os
from pathlib import Path

def _env_int(name, default):
    """读取整数型环境变量。

    用法：
        _env_int("BATCH_SIZE", 16)

    如果环境变量存在，就把它转成 int；如果不存在，就返回 default。
    这样云端训练时可以用 `BATCH_SIZE=32 python train.py` 临时覆盖配置。
    """
    value = os.environ.get(name)
    return int(value) if value else default

def get_config():
    """返回训练、验证、推理共同使用的配置字典。

    用法：
        config = get_config()

    默认值写在代码里，常用训练参数也支持通过环境变量覆盖：
    BATCH_SIZE、NUM_EPOCHS、LEARNING_RATE、SEQ_LEN、D_MODEL、
    MODEL_FOLDER、PRELOAD、EXPERIMENT_NAME、CHECKPOINT_INTERVAL。
    """
    return{
        "batch_size":_env_int("BATCH_SIZE",16),
        "num_epochs":_env_int("NUM_EPOCHS",20),
        "learning_rate":float(os.environ.get("LEARNING_RATE",1e-4)),
        "seq_len":_env_int("SEQ_LEN",256),
        "d_model":_env_int("D_MODEL",512),
        "dataset_name":"Helsinki-NLP/opus-100",
        "dataset_config":"en-zh",
        "dataset_disk_path":os.environ.get("DATASET_DISK_PATH"),
        "lang_src":"zh",
        "lang_tgt":"en",
        "model_folder":os.environ.get("MODEL_FOLDER","weights"),
        "model_basename":"tmodel_",
        "preload":os.environ.get("PRELOAD") or None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":os.environ.get("EXPERIMENT_NAME","runs/tmodel"),
        "checkpoint_interval":_env_int("CHECKPOINT_INTERVAL",100),
        "progress_log_interval_seconds":_env_int("PROGRESS_LOG_INTERVAL_SECONDS",1800),
    }

def get_weights_file_path(config,epoch:str):
    """根据配置和 epoch 名称生成 checkpoint 文件路径。

    用法：
        get_weights_file_path(config, "04")      -> weights/tmodel_04.pt
        get_weights_file_path(config, "latest")  -> weights/tmodel_latest.pt

    epoch 可以是具体轮次，也可以是 latest 这种特殊名称。
    """
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path(model_folder) / model_filename)
