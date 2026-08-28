from pathlib import Path

def get_config():
    return{
        "batch_size":16,
        "num_epochs":20,
        "learning_rate":1e-4,
        "seq_len":256,
        "d_model":512,
        "dataset_name":"Helsinki-NLP/opus-100",
        "dataset_config":"en-zh",
        "lang_src":"zh",
        "lang_tgt":"en",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }

def get_weights_file_path(config,epoch:str):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path(model_folder) / model_filename)
