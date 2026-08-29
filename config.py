import os
from pathlib import Path

def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default

def get_config():
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
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path(model_folder) / model_filename)
