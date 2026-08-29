import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import warnings
import signal
import time

from config import get_config,get_weights_file_path

from dataset import Bilingualdataset
from model import build_transformer

from datasets import load_dataset,load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split, Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Split(pattern='',behavior='isolated') if lang == 'zh' else Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def is_valid_length(item,tokenizer_src,tokenizer_tgt,config):
    src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    return len(src_ids)<=config['seq_len']-2 and len(tgt_ids)<=config['seq_len']-1

def get_ds(config):
    dataset_disk_path=config.get('dataset_disk_path')
    if dataset_disk_path and Path(dataset_disk_path).exists():
        ds_raw=load_from_disk(dataset_disk_path)
    else:
        ds_raw=load_dataset(config['dataset_name'],config['dataset_config'],split='train')

    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
    ds_raw=ds_raw.filter(lambda item:is_valid_length(item,tokenizer_src,tokenizer_tgt,config))

    train_ds_size=int(0.9*len(ds_raw))
    valid_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,valid_ds_raw=random_split(ds_raw,[train_ds_size,valid_ds_size])

    train_ds=Bilingualdataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    valid_ds=Bilingualdataset(valid_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    print(f'train_size={len(train_ds)}')
    print(f'valid_size={len(valid_ds)}')

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    valid_dataloader=DataLoader(valid_ds,batch_size=1,shuffle=True)

    return train_dataloader,valid_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def save_checkpoint(config,epoch,model,optimizer,global_step,epoch_name,epoch_completed):
    model_filename=get_weights_file_path(config,epoch_name)
    tmp_model_filename=f'{model_filename}.tmp'
    Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':epoch,
        'epoch_completed':epoch_completed,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'global_step':global_step,
    },tmp_model_filename)
    Path(tmp_model_filename).replace(model_filename)
    print(f'Saved checkpoint: {model_filename}', flush=True)

def format_duration(seconds):
    seconds=int(seconds)
    hours, seconds=divmod(seconds,3600)
    minutes, seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader,valid_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-9)


    initial_epoch=0
    global_step=0
    if config['preload']:
        model_filename=get_weights_file_path(config,config['preload'])
        print(f'Preloading {model_filename}')
        state=torch.load(model_filename,map_location=device)
        initial_epoch=state['epoch']+1 if state.get('epoch_completed',True) else state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    def handle_stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    current_epoch=initial_epoch
    train_started_at=time.monotonic()
    last_progress_log_at=train_started_at
    progress_log_interval=config.get('progress_log_interval_seconds',0)
    try:
        save_checkpoint(config,current_epoch,model,optimizer,global_step,'latest',False)
        for epoch in range(initial_epoch,config['num_epochs']):
            current_epoch=epoch
            model.train()
            batch_iterator=tqdm(train_dataloader,desc=f'Processing epoch {epoch:02d}')
            total_batches=len(train_dataloader)
            for batch_index,batch in enumerate(batch_iterator,start=1):
                encoder_input=batch['encoder_input'].to(device)
                decoder_input=batch['decoder_input'].to(device)
                encoder_mask=batch['encoder_mask'].to(device)
                decoder_mask=batch['decoder_mask'].to(device)

                encoder_output=model.encode(encoder_input,encoder_mask)
                decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
                proj_output=model.project(decoder_output)

                label=batch['label'].to(device)

                loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
                batch_iterator.set_postfix({f"loss:":f"{loss.item():6.3f}"})

                writer.add_scalar('loss',loss.item(),global_step=global_step)
                writer.flush()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step+=1

                checkpoint_interval=config.get('checkpoint_interval',0)
                if checkpoint_interval and global_step % checkpoint_interval == 0:
                    save_checkpoint(config,epoch,model,optimizer,global_step,'latest',False)

                now=time.monotonic()
                if progress_log_interval and now - last_progress_log_at >= progress_log_interval:
                    last_progress_log_at=now
                    percent=100*batch_index/total_batches
                    latest_checkpoint=get_weights_file_path(config,'latest')
                    print(
                        'Progress: '
                        f'epoch={epoch:02d}/{config["num_epochs"]-1:02d} '
                        f'batch={batch_index}/{total_batches} ({percent:.2f}%) '
                        f'global_step={global_step} '
                        f'loss={loss.item():.4f} '
                        f'elapsed={format_duration(now-train_started_at)} '
                        f'latest_checkpoint={latest_checkpoint}',
                        flush=True
                    )

            save_checkpoint(config,epoch,model,optimizer,global_step,f'{epoch:02d}',True)
            save_checkpoint(config,epoch,model,optimizer,global_step,'latest',True)
    except KeyboardInterrupt:
        print('Training interrupted; saving latest checkpoint before exit.', flush=True)
        save_checkpoint(config,current_epoch,model,optimizer,global_step,'latest',False)
        raise
    except Exception:
        print('Training failed; saving latest checkpoint for debugging/resume.', flush=True)
        save_checkpoint(config,current_epoch,model,optimizer,global_step,'latest',False)
        raise
    finally:
        writer.close()

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
