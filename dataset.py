import torch
import torch.nn as nn
from torch.utils.data import Dataset



class Bilingualdataset(Dataset):
    """把 HuggingFace 双语样本转换成 Transformer 训练用张量。

    用法：
        train_ds = Bilingualdataset(ds, tokenizer_src, tokenizer_tgt, "zh", "en", 256)

    输入样本格式需要包含：
        item["translation"]["zh"]  中文原文
        item["translation"]["en"]  英文目标句

    __getitem__ 会返回 encoder_input、decoder_input、mask、label 和原始文本。
    """
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)->None:
        """保存数据集、分词器、语言代码、最大序列长度和特殊 token id。"""
        super().__init__()

        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len

        self.src_sos_token_id=tokenizer_src.token_to_id('[SOS]')
        self.src_eos_token_id=tokenizer_src.token_to_id('[EOS]')
        self.src_pad_token_id=tokenizer_src.token_to_id('[PAD]')
        self.tgt_sos_token_id=tokenizer_tgt.token_to_id('[SOS]')
        self.tgt_eos_token_id=tokenizer_tgt.token_to_id('[EOS]')
        self.tgt_pad_token_id=tokenizer_tgt.token_to_id('[PAD]')

    def __len__(self):
        """返回样本数量，供 DataLoader 计算总 batch 数。"""
        return len(self.ds)

    def __getitem__(self, index):
        """取一条双语样本，并组装成模型训练需要的输入。

        encoder_input:
            [SOS] + 源语言 token + [EOS] + [PAD]...

        decoder_input:
            [SOS] + 目标语言 token + [PAD]...

        label:
            目标语言 token + [EOS] + [PAD]...

        encoder_mask 用于屏蔽源句 padding；decoder_mask 同时屏蔽 padding 和未来 token。
        """
        src_target_pair=self.ds[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens=self.tokenizer_src.encode(src_text).ids
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens=self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_tokens=self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError('Sentence is too long')

        encoder_input=torch.cat(
            [
                torch.tensor([self.src_sos_token_id],dtype=torch.int64),
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                torch.tensor([self.src_eos_token_id],dtype=torch.int64),
                torch.full((enc_num_padding_tokens,),self.src_pad_token_id,dtype=torch.int64)
            ]
        )

        decoder_input=torch.cat(
            [
                torch.tensor([self.tgt_sos_token_id],dtype=torch.int64),
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.full((dec_num_padding_tokens,),self.tgt_pad_token_id,dtype=torch.int64)
            ]
        )

        label=torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.tgt_eos_token_id],dtype=torch.int64),
                torch.full((dec_num_padding_tokens,),self.tgt_pad_token_id,dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return{
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input!=self.src_pad_token_id).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(decoder_input!=self.tgt_pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text
        }

def causal_mask(size):
    """生成 decoder 自注意力的因果 mask，防止模型偷看未来 token。

    用法：
        mask = causal_mask(decoder_input.size(0))

    返回形状为 (1, size, size) 的布尔矩阵，下三角为 True，上三角为 False。
    """
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0
