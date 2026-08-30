import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from dataset import causal_mask
from model import build_transformer


def find_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    weights_dir = run_dir / "weights"
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            return path
        candidate = weights_dir / checkpoint
        if candidate.exists():
            return candidate
        if not checkpoint.endswith(".pt"):
            candidate = weights_dir / f"tmodel_{checkpoint}.pt"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    numbered = sorted(weights_dir.glob("tmodel_[0-9][0-9].pt"))
    if numbered:
        return numbered[-1]

    latest = weights_dir / "tmodel_latest.pt"
    if latest.exists():
        return latest

    raise FileNotFoundError(f"No checkpoint found in {weights_dir}")


def load_tokenizers(run_dir: Path) -> tuple[Tokenizer, Tokenizer]:
    src_path = run_dir / "tokenizer_zh.json"
    tgt_path = run_dir / "tokenizer_en.json"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError("Expected tokenizer_zh.json and tokenizer_en.json in the run directory.")
    return Tokenizer.from_file(str(src_path)), Tokenizer.from_file(str(tgt_path))


def make_source_tensor(text: str, tokenizer_src: Tokenizer, seq_len: int, device: torch.device):
    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    pad_id = tokenizer_src.token_to_id("[PAD]")
    token_ids = tokenizer_src.encode(text).ids

    if len(token_ids) > seq_len - 2:
        raise ValueError(f"Input is too long: {len(token_ids)} tokens, max is {seq_len - 2}.")

    padding = seq_len - len(token_ids) - 2
    source = torch.cat(
        [
            torch.tensor([sos_id], dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([eos_id], dtype=torch.int64),
            torch.full((padding,), pad_id, dtype=torch.int64),
        ]
    ).unsqueeze(0).to(device)
    source_mask = (source != pad_id).unsqueeze(0).int()
    return source, source_mask


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    special_ids = {
        tokenizer.token_to_id("[SOS]"),
        tokenizer.token_to_id("[EOS]"),
        tokenizer.token_to_id("[PAD]"),
    }
    clean_ids = [token_id for token_id in ids if token_id not in special_ids]
    return tokenizer.decode(clean_ids).strip()


@torch.no_grad()
def translate(model, tokenizer_src, tokenizer_tgt, text: str, seq_len: int, device: torch.device) -> str:
    source, source_mask = make_source_tensor(text, tokenizer_src, seq_len, device)
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    pad_id = tokenizer_tgt.token_to_id("[PAD]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_id]], dtype=torch.int64, device=device)

    while decoder_input.size(1) < seq_len:
        decoder_mask = (decoder_input != pad_id).unsqueeze(0).int()
        decoder_mask = decoder_mask & causal_mask(decoder_input.size(1)).type_as(decoder_mask).to(device)

        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(decoder_output[:, -1])
        next_id = torch.argmax(logits, dim=-1).item()
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_id]], dtype=torch.int64, device=device)],
            dim=1,
        )
        if next_id == eos_id:
            break

    return decode_ids(tokenizer_tgt, decoder_input.squeeze(0).detach().cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description="Translate Chinese text with a trained checkpoint.")
    parser.add_argument("--run-dir", default="trained_models/jupyter-iz8apfja-5g1b3hr1")
    parser.add_argument("--checkpoint", default=None, help="Path, filename, or epoch id such as 04/latest.")
    parser.add_argument("--text", default=None, help="Chinese text to translate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = get_config()
    tokenizer_src, tokenizer_tgt = load_tokenizers(run_dir)
    checkpoint_path = find_checkpoint(run_dir, args.checkpoint)

    device = torch.device(args.device)
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")

    if args.text:
        print(translate(model, tokenizer_src, tokenizer_tgt, args.text, config["seq_len"], device))
        return

    print("Enter Chinese text. Press Ctrl+C or submit an empty line to exit.")
    while True:
        text = input("zh> ").strip()
        if not text:
            break
        print(f"en> {translate(model, tokenizer_src, tokenizer_tgt, text, config['seq_len'], device)}")


if __name__ == "__main__":
    main()
