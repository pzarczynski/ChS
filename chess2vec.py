import csv
import functools
import gzip
from functools import partial
from itertools import islice, zip_longest

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

hex2int = partial(int, base=16)


def int2hex(x):
    return f"{x:x}"


def fill_uneven(x, value=None):
    return zip(*zip_longest(*x, fillvalue=value))


def strip(x: str):
    return x.strip()


def games(f):
    with open(f, encoding="UTF-8") as pgn:
        while game := chess.pgn.read_game(pgn):
            yield game


def positions(game):
    board = game.board()

    for m in game.mainline_moves():
        yield board
        board.push(m)

    yield board


def bitboards(pos):
    return [
        pos.occupied_co[chess.BLACK],
        pos.occupied_co[chess.WHITE],
    ], [
        pos.pawns,
        pos.knights,
        pos.bishops,
        pos.rooks,
        pos.queens,
        pos.kings,
    ]


def bb_flip(b, orientation=chess.WHITE):
    return b if orientation else chess.flip_vertical(b)


def pos_toint(pos):
    colors, pieces = bitboards(pos)
    colors = [colors[not pos.turn], colors[pos.turn]]

    bbs = [bb_flip(p & c, pos.turn) for p in pieces for c in colors]

    return functools.reduce(lambda x, y: x << 64 | y, bbs)


def pos_tosparse(pos):
    yield from chess.scan_forward(pos_toint(pos))


def pos_stream(f):
    for g in games(f):
        for pos in positions(g):
            yield pos


def pgn_prepare(pgn, base_path, with_labels=False):
    if with_labels:
        f_labels = gzip.open(base_path + "_labels", "wt")

    with gzip.open(base_path + "_indices", "wt", newline="") as f_indices:
        writer = csv.writer(f_indices, delimiter=" ")

        for pos in tqdm(pos_stream(pgn), desc=pgn):
            writer.writerow(map(int2hex, pos_tosparse(pos)))

            if with_labels:
                f_labels.write(pos.fen() + "\n")

    if with_labels:
        f_labels.close()


def prepared_stream(f_indices, f_labels=None):
    reader = csv.reader(f_indices, delimiter=" ")
    indices = (list(map(hex2int, idx)) for idx in reader)

    yield from zip(indices, map(strip, f_labels)) if f_labels else indices


def load_prepared(f_indices, f_labels=None, count=None):
    if f_labels:
        indices, labels = prepared_stream(f_indices, f_labels)
    else:
        indices = prepared_stream(f_indices)

    evened = fill_uneven(islice(indices, count), -1)
    mat = np.fromiter(evened, dtype=np.dtype((np.int16, 32)))

    return (mat, labels) if f_labels else mat
