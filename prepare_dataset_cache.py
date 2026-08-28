from pathlib import Path

from datasets import load_dataset

from config import get_config


def main():
    config = get_config()
    output_dir = Path("dataset_cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        config["dataset_name"],
        config["dataset_config"],
        split="train",
    )
    dataset.save_to_disk(str(output_dir))
    print(f"Saved dataset cache to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
