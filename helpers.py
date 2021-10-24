import fnmatch
import os
import random
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from chess import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                   Board, Move)


def replace_tags(fen):
    fen_split = fen.split(" ")
    fen_format = fen_split[0]
    fen_format = fen_format.replace("2", "11")
    fen_format = fen_format.replace("3", "111")
    fen_format = fen_format.replace("4", "1111")
    fen_format = fen_format.replace("5", "11111")
    fen_format = fen_format.replace("6", "111111")
    fen_format = fen_format.replace("7", "1111111")
    fen_format = fen_format.replace("8", "11111111")
    fen_format = fen_format.replace("/", "")
    for i in range(len(fen_split)):
        if i > 0 and fen.split(" ")[i] != '':
            fen_format += " " + fen_split[i]
    return fen_format

value = {'p': 1, 'P': 1, 'r': 5, 'R': 5, 'n': 3, 'N': 3, 'b': 3, 'B': 3, 'q': 10, 'Q': 10, 'k': 100, 'K': 100, '1': 0}
def to_planes(board_pieces: list):
    # All pieces plane
    board_all = [value[val] for val in board_pieces]
    board_all = np.reshape(board_all, (8, 8))
    # Only spaces plane
    board_blank = [int(val == '1') for val in board_pieces]
    board_blank = np.reshape(board_blank, (8, 8))
    # Only white plane
    board_white = [int(val.isupper()) for val in board_pieces]
    board_white = np.reshape(board_white, (8, 8))
    # Only black plane
    board_black = [int(not val.isupper() and val != '1') for val in board_pieces]
    board_black = np.reshape(board_black, (8, 8))

    pawns = [int(val == 'p') for val in board_pieces]
    pawns = np.reshape(pawns, (8, 8))
    rooks = [int(val == 'r') for val in board_pieces]
    rooks = np.reshape(rooks, (8, 8))
    nights = [int(val == 'n') for val in board_pieces]
    nights = np.reshape(nights, (8, 8))
    bishops = [int(val == 'b') for val in board_pieces]
    bishops = np.reshape(bishops, (8, 8))
    queens = [int(val == 'q') for val in board_pieces]
    queens = np.reshape(queens, (8, 8))
    kings = [int(val == 'k') for val in board_pieces]
    kings = np.reshape(kings, (8, 8))
    pawns2 = [int(val == 'P') for val in board_pieces]
    pawns2 = np.reshape(pawns2, (8, 8))
    rooks2 = [int(val == 'R') for val in board_pieces]
    rooks2 = np.reshape(rooks2, (8, 8))
    nights2 = [int(val == 'N') for val in board_pieces]
    nights2 = np.reshape(nights2, (8, 8))
    bishops2 = [int(val == 'B') for val in board_pieces]
    bishops2 = np.reshape(bishops2, (8, 8))
    queens2 = [int(val == 'Q') for val in board_pieces]
    queens2 = np.reshape(queens2, (8, 8))
    kings2 = [int(val == 'K') for val in board_pieces]
    kings2 = np.reshape(kings2, (8, 8))

    planes = np.stack(( np.copy(board_blank),
                        np.copy(pawns),
                        np.copy(rooks),
                        np.copy(nights),
                        np.copy(bishops),
                        np.copy(queens),
                        np.copy(kings),
                        np.copy(pawns2),
                        np.copy(rooks2),
                        np.copy(nights2),
                        np.copy(bishops2),
                        np.copy(queens2),
                        np.copy(kings2)
                        ))

    # print(planes)
    # raise EnvironmentError
    return planes


def reformat(game):
    board_state0 = None
    board_state1 = None
    board_state2 = replace_tags(game.fen())
    try:
        game.pop()
        game.pop()
        board_state1 = replace_tags(game.fen())
        game.pop()
        game.pop()
        board_state0 = replace_tags(game.fen())
    except IndexError:
        if board_state1 is None:
            board_state1 = replace_tags(game.fen())
        if board_state0 is None:
            board_state0 = replace_tags(game.fen())
    except Exception as e:
        print(e)
        return
    # print(board)
    # print(label_state)


    # All pieces plane
    board_pieces0 = list(board_state0.split(" ")[0])
    game_feat0 = to_planes(board_pieces0)
    board_pieces1 = list(board_state1.split(" ")[0])
    game_feat1 = to_planes(board_pieces1)
    board_pieces2 = list(board_state2.split(" ")[0])
    game_feat2 = to_planes(board_pieces2)
    # One-hot integer plane current player turn
    current_player = board_state2.split(" ")[1]
    current_player = np.full((8, 8), int(current_player == 'w'), dtype=int)
    # One-hot integer plane extra data
    extra = board_state2.split(" ")[4]
    extra = np.full((8, 8), int(extra), dtype=int)

    planes = np.concatenate((np.copy(game_feat0),
                        np.copy(game_feat1),
                        np.copy(game_feat2),
                        np.copy([current_player]),
                        np.copy([extra])),
                        axis=0
                        )
    planes = np.reshape(planes, (-1, 8, 8))
    return planes

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def read_labels(filename):
    labels_array = []
    with open(filename) as f:
        lines = str(f.readlines()[0]).split(" ")
        for label in lines:
            if(label != " " and label != '\n'):
                labels_array.append(label)
    return labels_array


def _read_text(filename, batch_size):
    with open(filename) as f:
        return random.sample(f.readlines(), batch_size)


def generate_batch(batch_size, directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    random.shuffle(files)
    for filename in files:
        text = _read_text(filename, batch_size)
        yield text

def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)

def to_planes(board: Board):
    pieces = np.array([
        board.pieces_mask(PAWN, WHITE),
        board.pieces_mask(ROOK, WHITE),
        board.pieces_mask(KNIGHT, WHITE),
        board.pieces_mask(BISHOP, WHITE),
        board.pieces_mask(QUEEN, WHITE),
        board.pieces_mask(KING, WHITE),
        board.pieces_mask(PAWN, BLACK),
        board.pieces_mask(ROOK, BLACK),
        board.pieces_mask(KNIGHT, BLACK),
        board.pieces_mask(BISHOP, BLACK),
        board.pieces_mask(QUEEN, BLACK),
        board.pieces_mask(KING, BLACK),
    ])
    pieces_planes = bitboards_to_array(pieces)
    blank = np.ones((8, 8))
    for plane in pieces_planes:
        blank -= plane
    # Only spaces plane

    planes = np.concatenate(([blank], pieces_planes), axis=0)
    return np.reshape(planes, (-1, 8, 8))


def choose_move(model, planes_0, planes_1, planes_2, player, extra, board, label_to_idx):
    planes = np.concatenate((planes_0,
                             planes_1,
                             planes_2,
                             player,
                             extra),
                            axis=0
                            )
    try:
        planes = torch.FloatTensor([planes]).cuda()
        preds = model(planes)
    except:
        print(planes)
        raise
    legals = torch.zeros(1965)
    for move in board.legal_moves:
        try:
            legals[label_to_idx[move.uci()]] = 1
        except KeyError:
            pass

    choices = torch.sum(legals).item()
    masked_preds = preds.clone().detach() * legals.cuda()

    top = torch.topk(masked_preds[0], 3)
    if torch.count_nonzero(top.values) > 0:
        m = Categorical(probs=top.values)
        move_idx = top.indices[m.sample().item()].item()
    else:
        try:
            m = Categorical(probs=legals)
            move_idx = m.sample().item()
            return preds[0], move_idx, choices
        except:
            raise

    return preds[0], move_idx, choices