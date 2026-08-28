import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from dataset import causal_mask
from train import get_ds,get_model


def latest_checkpoint(model_folder):
    weights = sorted(Path(model_folder).glob('*.pt'))
    return weights[-1] if weights else None


def load_model(config,tokenizer_src,tokenizer_tgt,device,weights_path):
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    state = torch.load(weights_path,map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def greedy_decode(model,source,source_mask,tokenizer_tgt,max_len,device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')

    encoder_output = model.encode(source,source_mask)
    decoder_input = torch.empty(1,1,dtype=source.dtype,device=device).fill_(sos_idx)

    while decoder_input.size(1) < max_len:
        decoder_mask = (decoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = decoder_mask & causal_mask(decoder_input.size(1)).type_as(decoder_mask).to(device)

        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        logits = model.project(out[:,-1])
        next_word = torch.argmax(logits,dim=-1).item()

        decoder_input = torch.cat(
            [decoder_input,torch.empty(1,1,dtype=source.dtype,device=device).fill_(next_word)],
            dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0).tolist()


def run_validation(model,validation_dataloader,tokenizer_tgt,device,max_len,max_batches,num_examples):
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx,reduction='sum')

    total_loss = 0.0
    total_tokens = 0
    examples = []

    with torch.no_grad():
        iterator = enumerate(validation_dataloader)
        if max_batches > 0:
            iterator = zip(range(max_batches),validation_dataloader)

        for batch_idx,batch in tqdm(iterator,desc='Validation'):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1,tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            total_loss += loss.item()
            total_tokens += (label != pad_idx).sum().item()

            if len(examples) < num_examples and encoder_input.size(0) == 1:
                predicted_ids = greedy_decode(model,encoder_input,encoder_mask,tokenizer_tgt,max_len,device)
                examples.append((
                    batch['src_text'][0],
                    batch['tgt_text'][0],
                    tokenizer_tgt.decode(predicted_ids)
                ))

            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    avg_loss = total_loss / max(total_tokens,1)
    return avg_loss,math.exp(avg_loss),examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default=None)
    parser.add_argument('--max-batches',type=int,default=100)
    parser.add_argument('--num-examples',type=int,default=3)
    parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = get_config()
    _,validation_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)

    weights_path = Path(args.weights) if args.weights else latest_checkpoint(config['model_folder'])
    if weights_path is None:
        raise FileNotFoundError(f"No checkpoint found in {config['model_folder']}. Train at least one epoch first.")

    device = torch.device(args.device)
    model = load_model(config,tokenizer_src,tokenizer_tgt,device,weights_path)

    avg_loss,ppl,examples = run_validation(
        model,
        validation_dataloader,
        tokenizer_tgt,
        device,
        config['seq_len'],
        args.max_batches,
        args.num_examples
    )

    print(f'weights={weights_path}')
    print(f'validation_loss={avg_loss:.4f}')
    print(f'perplexity={ppl:.4f}')

    for idx,(src,expected,predicted) in enumerate(examples,1):
        print('-' * 80)
        print(f'example {idx}')
        print(f'source: {src}')
        print(f'expected: {expected}')
        print(f'predicted: {predicted}')


if __name__ == '__main__':
    main()
