import os

from fire import Fire

from chess2vec import pgn_prepare


def main(source_dir=".", target_dir=".", with_labels: bool = False):
    for file in os.listdir(source_dir):
        if file.endswith(".pgn"):
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file.removesuffix(".pgn"))

            pgn_prepare(source_path, target_path, with_labels=with_labels)


if __name__ == "__main__":
    Fire(main)
