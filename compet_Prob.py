import weights as wgts
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from chess import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                   Board, Move, Outcome, Termination)
from tqdm import tqdm
from models import ChessModelBase, ChessModel
from helpers import to_planes, read_labels, choose_move
from compet_AB import basicAB

LABELS_FILE = './labels/labels.txt'
MODEL_PATH = '/media/tergaim/Data/chess_data/experiment_rl/base.pth'

FEATURE_PLANES = 13
LABEL_SIZE = 1965

available_labels = []
label_to_idx = {}


class basicProb:
    def __init__(self, model, threshold, label_to_idx):
        self.model = model
        self.threshold = threshold
        self.label_to_idx = label_to_idx

        self.white_player = [np.full((8, 8), 1, dtype=int)]
        self.black_player = [np.full((8, 8), 0, dtype=int)]

    def runProb(self, board: Board, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B, depth=8):
        extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
        if board.turn:
            preds, _, _ = choose_move(self.model, planes_0W, planes_1W, planes_2W, self.white_player, extra, board, self.label_to_idx)
        else:
            preds, _, _ = choose_move(self.model, planes_0B, planes_1B, planes_2B, self.black_player, extra, board, self.label_to_idx)

        self.turn = board.turn
        alpha = 0
        beta = 0
        bestValue = -100000
        bestmove = None
        # print(list(board.legal_moves))
        for move in board.legal_moves:
            board.push(move)
            if board.turn:
                value = self.minProb(board, depth, preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_1W, planes_2W, to_planes(board), planes_0B, planes_1B, planes_2B)
            else:
                value = self.minProb(board, depth, preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_0W, planes_1W, planes_2W, planes_1B, planes_2B, to_planes(board))
            board.pop()
            if value > bestValue:
                bestValue = value
                bestmove = move
                # print(f"Pondering {move} with score {bestValue}")
        # print(bestmove)
        return bestmove

    def evalPos(self, board: Board):
        piece_map = board.piece_map()
        score = 0

        for s, p in piece_map.items():
            s = 63 - s
            piece_value = wgts.piece[p.piece_type]  # + wgts.pst[p.piece_type][s]
            allied = 1 if p.color == self.turn else -1
            score += allied * piece_value
        return score

    def maxProb(self, board: Board, depth, alpha, beta, prob_node, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B):
        if depth == 0 or prob_node < self.threshold:
            return self.evalPos(board)

        minValue = -1000000
        extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
        if board.turn:
            preds, _, _ = choose_move(self.model, planes_0W, planes_1W, planes_2W, self.white_player, extra, board, self.label_to_idx)
        else:
            preds, _, _ = choose_move(self.model, planes_0B, planes_1B, planes_2B, self.black_player, extra, board, self.label_to_idx)

        for move in board.legal_moves:
            board.push(move)
            end = board.outcome()
            if end is None:
                if self.turn:
                    minValue = max(minValue, self.minAB(board, depth-1, alpha, beta, prob_node *
                                                        preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_1W, planes_2W, to_planes(board), planes_0B, planes_1B, planes_2B))
                else:
                    minValue = max(minValue, self.minAB(board, depth-1, alpha, beta, prob_node *
                                                        preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_0W, planes_1W, planes_2W, planes_0B.copy(), planes_1B.copy(), to_planes(board)))
            elif end.winner == self.turn:
                minValue = -60000
            elif end.winner == 1 - int(self.turn):
                minValue = 60000
            else:
                minValue = 0
            board.pop()

            if minValue >= beta:
                return minValue
            alpha = max(alpha, minValue)

        return minValue

    def minProb(self, board: Board, depth, alpha, beta, prob_node, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B):
        if depth == 0 or prob_node < self.threshold:
            return self.evalPos(board)

        maxValue = 1000000
        extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
        if board.turn:
            preds, _, _ = choose_move(self.model, planes_0W, planes_1W, planes_2W, self.white_player, extra, board, self.label_to_idx)
        else:
            preds, _, _ = choose_move(self.model, planes_0B, planes_1B, planes_2B, self.black_player, extra, board, self.label_to_idx)

        for move in board.legal_moves:
            board.push(move)
            end = board.outcome()
            if end is None:
                if self.turn:
                    maxValue = min(maxValue, self.maxAB(board, depth-1, alpha, beta, prob_node *
                                                        preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_1W, planes_2W, to_planes(board), planes_0B, planes_1B, planes_2B))
                else:
                    maxValue = min(maxValue, self.maxAB(board, depth-1, alpha, beta, prob_node *
                                                        preds[self.label_to_idx[board.uci(move)]], alpha, beta, planes_0W, planes_1W, planes_2W, planes_1B, planes_2B, to_planes(board)))

            elif end.winner == self.turn:
                maxValue = 60000
            elif end.winner == 1 - int(self.turn):
                maxValue = -60000
            else:
                maxValue = 0
            board.pop()

            if maxValue <= alpha:
                return maxValue
            beta = min(beta, maxValue)

        return maxValue


def play_game_Prob(model, modeliswhite):
    player = basicProb(model, 0.001, label_to_idx)

    white_player = [np.full((8, 8), 1, dtype=int)]
    black_player = [np.full((8, 8), 0, dtype=int)]

    other = basicAB()

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

                board.push(player.runProb(board, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B))
            else:
                board.push(other.runAB(board))
            end = board.outcome()
            if end is not None:
                break

            if not modeliswhite:
                planes_0B = planes_1B
                planes_1B = planes_2B
                planes_2B = to_planes(board)

                board.push(player.runProb(board, planes_0W, planes_1W, planes_2W, planes_0B, planes_1B, planes_2B))
            else:
                board.push(other.runAB(board))
            end = board.outcome()

    except Exception as e:
        print(e)
        raise

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
        win = play_game_Prob(model, True)
        if win < 0:
            win = 0.5
            aborted += 1
        score_w += win
        win = play_game_Prob(model, False)
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
