import faiss
from fire import Fire


def main():
    index = faiss.IndexBinaryFlat(768)
    index.add()


if __name__ == "__main__":
    Fire(main)
