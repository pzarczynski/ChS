import io

import cairosvg
import chess
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_boards(fens, annot, *args, **kwargs):
    fig, axarr = plt.subplots(*args, **kwargs)

    for fen, t, ax in zip(fens, annot, axarr.ravel()):
        board = chess.Board(fen)

        svg = chess.svg.board(board, orientation=board.turn)

        png = cairosvg.svg2png(svg)
        img = Image.open(io.BytesIO(png))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(t), y=-0.15)

    plt.tight_layout()


def show_matrices(mats, cmaps, *args, figsize=None, **kwargs):
    fig, axarr = plt.subplots(*args, figsize=figsize)
    axarr = np.array([axarr])

    for mat, cm, ax in zip(mats, cmaps, axarr.ravel()):
        img = ax.matshow(mat, cmap=cm)
        plt.colorbar(img, ax=ax, ticks=[np.min(mat), np.max(mat)], **kwargs)


def sample(rng, n, k, *args):
    idx = rng.integers(0, n, k)
    return tuple(map(lambda x: x[idx], args))


def normalize_minmax(x):
    return -1 + 2 * (x - x.min(1)) / (x.max(1) - x.min(1))
