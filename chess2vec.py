import functools

import chess
import chess.pgn
import numpy as np


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
    bb = colors[pos.turn] << 64 | colors[not pos.turn]

    return functools.reduce(lambda x, y: x << 64 | bb_flip(y, pos.turn), pieces, bb)


def pos_tosparse(pos):
    yield from chess.scan_forward(pos_toint(pos))


def pos_stream(f):
    for g in games(f):
        for pos in positions(g):
            yield pos


# def accumulate_apply(func, x):
#     while x:
#         yield x
#         x = func(x)


# def shift64right(x):
#     return x >> 64


# def clip64(x):
#     return x & 0xFFFF_FFFF_FFFF_FFFF


# def int_asarray(x):
#     return np.fromiter(
#         map(clip64, accumulate_apply(shift64right, x)), dtype=np.uint64
#     )
