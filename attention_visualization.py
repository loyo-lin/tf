import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import get_config
from dataset import causal_mask
from train import get_ds,get_model


def latest_checkpoint(model_folder):
    """返回模型目录中按文件名排序后的最新 checkpoint。

    用法：
        weights_path = latest_checkpoint("weights")

    如果没有 .pt 文件，返回 None。
    """
    weights = sorted(Path(model_folder).glob('*.pt'))
    return weights[-1] if weights else None


def ids_to_tokens(tokenizer,ids,pad_id):
    """把 token id 列表转换成 token 字符串列表，并在遇到 PAD 后停止。

    用法：
        tokens = ids_to_tokens(tokenizer_src, encoder_input_ids, pad_id)

    可视化注意力时，x/y 轴标签就是这些 token。
    """
    tokens = []
    for token_id in ids:
        if token_id == pad_id:
            break
        tokens.append(tokenizer.id_to_token(int(token_id)))
    return tokens


def load_model(config,tokenizer_src,tokenizer_tgt,device,weights_path):
    """创建模型并加载训练好的 checkpoint。

    用法：
        model = load_model(config, tokenizer_src, tokenizer_tgt, device, "weights/tmodel_04.pt")

    加载后会切到 eval 模式，适合推理和注意力可视化。
    """
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    state = torch.load(weights_path,map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def greedy_decode(model,source,source_mask,tokenizer_tgt,max_len,device):
    """用贪心解码生成目标序列，并让模型保留最后一次注意力分数。

    用法：
        decoder_input = greedy_decode(model, source, source_mask, tokenizer_tgt, 256, device)

    mode=greedy 时使用它，适合观察模型自己生成译文时的注意力。
    """
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

    decoder_mask = (decoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int()
    decoder_mask = decoder_mask & causal_mask(decoder_input.size(1)).type_as(decoder_mask).to(device)
    encoder_output = model.encode(source,source_mask)
    model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
    return decoder_input


def save_heatmap(matrix,x_tokens,y_tokens,title,path):
    """把注意力矩阵保存成热力图图片。

    用法：
        save_heatmap(attention, src_tokens, tgt_tokens, "decoder cross-attention", out_path)

    x_tokens 会显示在横轴，y_tokens 会显示在纵轴。
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    width = max(8,min(24,0.45 * len(x_tokens) + 4))
    height = max(6,min(18,0.35 * len(y_tokens) + 3))
    fig,ax = plt.subplots(figsize=(width,height))
    image = ax.imshow(matrix,cmap='viridis',aspect='auto')
    ax.set_title(title)
    ax.set_xticks(range(len(x_tokens)))
    ax.set_yticks(range(len(y_tokens)))
    ax.set_xticklabels(x_tokens,rotation=80,ha='right',fontsize=8)
    ax.set_yticklabels(y_tokens,fontsize=8)
    fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    """命令行入口：加载模型、取一条验证样本并输出注意力图。

    用法：
        python attention_visualization.py --weights weights/tmodel_04.pt --index 0 --mode greedy

    常用参数：
        --index 选择第几条验证样本。
        --layer 选择 decoder 第几层，默认 -1 表示最后一层。
        --head 选择注意力头编号。
        --mode teacher 使用标准答案作为 decoder 输入，greedy 使用模型自己生成的译文。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default=None)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--layer',type=int,default=-1)
    parser.add_argument('--head',type=int,default=0)
    parser.add_argument('--mode',choices=['teacher','greedy'],default='teacher')
    parser.add_argument('--out-dir',type=str,default='attention_outputs')
    parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = get_config()
    _,validation_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)

    weights_path = Path(args.weights) if args.weights else latest_checkpoint(config['model_folder'])
    if weights_path is None:
        raise FileNotFoundError(f"No checkpoint found in {config['model_folder']}. Train at least one epoch first.")

    device = torch.device(args.device)
    model = load_model(config,tokenizer_src,tokenizer_tgt,device,weights_path)

    validation_iterator = iter(validation_dataloader)
    batch = next(validation_iterator)
    for _ in range(args.index):
        batch = next(validation_iterator)

    source = batch['encoder_input'].to(device)
    source_mask = batch['encoder_mask'].to(device)
    src_pad_id = tokenizer_src.token_to_id('[PAD]')
    tgt_pad_id = tokenizer_tgt.token_to_id('[PAD]')

    if args.mode == 'greedy':
        decoder_input = greedy_decode(model,source,source_mask,tokenizer_tgt,config['seq_len'],device)
    else:
        decoder_input = batch['decoder_input'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        with torch.no_grad():
            encoder_output = model.encode(source,source_mask)
            model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

    src_tokens = ids_to_tokens(tokenizer_src,batch['encoder_input'][0].tolist(),src_pad_id)
    tgt_tokens = ids_to_tokens(tokenizer_tgt,decoder_input[0].detach().cpu().tolist(),tgt_pad_id)

    decoder_layer = model.decoder.layers[args.layer]
    self_attention = decoder_layer.self_attention_block.attention_scores[0,args.head]
    cross_attention = decoder_layer.cross_attention_block.attention_scores[0,args.head]

    self_attention = self_attention[:len(tgt_tokens),:len(tgt_tokens)].detach().cpu().numpy()
    cross_attention = cross_attention[:len(tgt_tokens),:len(src_tokens)].detach().cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    self_path = out_dir / 'decoder_self_attention.png'
    cross_path = out_dir / 'decoder_cross_attention.png'

    save_heatmap(self_attention,tgt_tokens,tgt_tokens,'decoder self-attention',self_path)
    save_heatmap(cross_attention,src_tokens,tgt_tokens,'decoder cross-attention',cross_path)

    print(f'weights={weights_path}')
    print(f'source: {batch["src_text"][0]}')
    print(f'target: {batch["tgt_text"][0]}')
    if args.mode == 'greedy':
        print(f'predicted: {tokenizer_tgt.decode(decoder_input[0].detach().cpu().tolist())}')
    print(f'self_attention={self_path}')
    print(f'cross_attention={cross_path}')


if __name__ == '__main__':
    main()
