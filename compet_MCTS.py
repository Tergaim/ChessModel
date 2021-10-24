import chess
import numpy as np
import torch
from chess import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                   Board, Move, Outcome, Termination)
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from models import ChessModelBase, ChessModel
from helpers import to_planes, choose_move, read_labels
import random

LABELS_FILE = './labels/labels.txt'
MODEL_PATH = ''

FEATURE_PLANES = 13
LABEL_SIZE = 1965

available_labels = []
label_to_idx = {}


class basicMCTS():
    def __init__(self, model):
        self. model = model
        self.gamespermove = 40

    def imitate_game(self, board, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B):
        white_player = [np.full((8, 8), 1, dtype=int)]
        black_player = [np.full((8, 8), 0, dtype=int)]

        too_long = False
        end = None
        white = True
        counter = 0

        while counter < 100 and end is None:
            extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
            try:
                if white:
                    planes_0W = planes_1W
                    planes_1W = planes_2W
                    planes_2W = to_planes(board)
                    _, move_idx, _ = choose_move(self.model, planes_0W, planes_1W, planes_2W, white_player, extra, board)

                else:
                    planes_0B = planes_1B
                    planes_1B = planes_2B
                    planes_2B = to_planes(board)
                    _, move_idx, _ = choose_move(self.model, planes_0B, planes_1B, planes_2B, black_player, extra, board)
                board.push_uci(available_labels[move_idx])

            except Exception as e:  # happens when no legal move is possible but wasn't detected by board.outcome()
                counter = 1000  # end the game in a draw
            counter += 1
            white = not white
            end = board.outcome()

        win = 0.5
        if end is not None:
            win = 1 if end.winner else 0
        return win

    def runMCTS(self, board, planes4, planes2, planes0, planes5, planes3):
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        best_score = -1
        select_from = []

        for move in moves:
            board_copy = board.copy()
            board_copy.push(move)
            planes1 = to_planes(board_copy)

            score = 0
            for _ in range(self.gamespermove):
                if board.turn == BLACK:
                    score += 1 - self.imitate_game(board_copy, planes4, planes2, planes0, planes5, planes3, planes1)
                else:
                    score += self.imitate_game(board_copy, planes5, planes3, planes1, planes4, planes2, planes0)
            if score == best_score:
                select_from.append(move)
            elif score > best_score:
                best_score = score
                select_from = [move]

        return random.choice(select_from)


def play_game_MCTS(model, modeliswhite):
    other = basicMCTS(model)
    white_player = [np.full((8, 8), 1, dtype=int)]
    black_player = [np.full((8, 8), 0, dtype=int)]

    board = Board()
    planes_0W = to_planes(board)
    planes_1W = planes_0W
    planes_2W = planes_0W
    planes_0B = planes_0W
    planes_1B = planes_0W
    planes_2B = planes_0W

    too_long = False
    end = None
    try:
        while end is None and not too_long:
            extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
            if modeliswhite:
                planes_0W = planes_1W
                planes_1W = planes_2W
                planes_2W = to_planes(board)

                _, move_idx, _ = choose_move(model, planes_0W, planes_1W, planes_2W, white_player, extra, board, label_to_idx)

                board.push_uci(available_labels[move_idx])
            else:
                board.push(other.runMCTS(board, planes_0W, planes_1W, planes_2W, planes_1B, planes_2B))
            end = board.outcome()
            if end is not None:
                break

            if not modeliswhite:
                planes_0B = planes_1B
                planes_1B = planes_2B
                planes_2B = to_planes(board)

                _, move_idx, _ = choose_move(model, planes_0B, planes_1B, planes_2B, white_player, extra, board, label_to_idx)

                board.push_uci(available_labels[move_idx])
            else:
                board.push(other.runMCTS(board, planes_0B, planes_1B, planes_2B, planes_1W, planes_2W))
            end = board.outcome()

            # if len(board.move_stack) > 300:
            #     too_long = True
    except Exception as e:
        print(e)
        raise

    # if too_long:
    #     return -1
    # print(end)

    win = 0.5
    if end.winner == WHITE:
        win = 1
    elif end.winner == BLACK:
        win = 0

    return win


def test(model):
    score_w = 0.0
    score_b = 0.0
    aborted = 0
    draw = 0
    for n in tqdm(range(50)):
        win = play_game_MCTS(model, True)
        if win < 0:
            win = 0.5
            aborted += 1
        score_w += win
        win = play_game_MCTS(model, False)
        if win == 0.5:
            draw += 1
        if win < 0:
            win = 0.5
            aborted += 1
        score_b += 1-win
        print(f'Score after {(n+1)*2} games: {(score_w+score_b)*50/(n+1)}')
    # print(f'score = {score}')
    print(f"'Test/Score_All', {(score_w + score_b)}")
    print(f"'Test/Draws', {draw}")
    print(f"'Test/Aborted', {aborted}")
    return (score_w + score_b)


def main():
    torch.cuda.set_device(0)
    global available_labels
    global label_to_idx
    available_labels = read_labels(LABELS_FILE)
    for i, l in enumerate(available_labels):
        label_to_idx[l] = i
    # model = ChessModelBase(3*FEATURE_PLANES+2).cuda()
    model = ChessModel(3*FEATURE_PLANES+2).cuda()

    try:
        model.load_state_dict(torch.load(MODEL_PATH), strict=False)
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    test(model)


if __name__ == '__main__':
    main()
