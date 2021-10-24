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

LABELS_FILE = './labels/labels.txt'
MODEL_PATH = ''

FEATURE_PLANES = 13
LABEL_SIZE = 1965

available_labels = []
label_to_idx = {}

class basicAB:
    def runAB(self, board: Board, depth=4):
        self.turn = board.turn
        alpha = 0
        beta = 0
        bestValue = -1000
        bestmove = None
        for move in board.legal_moves:
            board.push(move)
            value = self.minAB(board, depth, alpha, beta)
            board.pop()
            if value > bestValue:
                bestValue = value
                bestmove = move
                # print(f"Pondering {move} with score {bestValue}")
        return bestmove

    def evalPos(self, board: Board):
        piece_map = board.piece_map()
        score = 0

        for s, p in piece_map.items():
            s = 63 - s
            piece_value = wgts.piece[p.piece_type] #+ wgts.pst[p.piece_type][s]
            allied = 1 if p.color == self.turn else -1
            score += allied * piece_value
        return score

    def maxAB(self, board: Board, depth: int, alpha: float, beta: float):
        if depth == 0:
            return self.evalPos(board)

        minValue = -1000000
        for move in board.legal_moves:
            board.push(move)
            end = board.outcome()
            if end is None:
                minValue = max(minValue, self.minAB(board, depth-1, alpha, beta))
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

    def minAB(self, board: Board, depth: int, alpha: float, beta: float):
        if depth == 0:
            return self.evalPos(board)

        maxValue = 1000000
        for move in board.legal_moves:
            board.push(move)
            end = board.outcome()
            if end is None:
                maxValue = min(maxValue, self.maxAB(board, depth-1, alpha, beta))
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

def play_game_AB(model, modeliswhite):
    if modeliswhite:
        w = model
    else:
        b = model
    
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

                _, move_idx, _ = choose_move(w, planes_0W, planes_1W, planes_2W, white_player, extra, board, label_to_idx)

                board.push_uci(available_labels[move_idx])
            else:
                board.push(other.runAB(board))
            end = board.outcome()
            if end is not None:
                break

            if not modeliswhite:
                planes_0B = planes_1B
                planes_1B = planes_2B
                planes_2B = to_planes(board)

                _, move_idx, _ = choose_move(b, planes_0B, planes_1B, planes_2B, white_player, extra, board, label_to_idx)

                board.push_uci(available_labels[move_idx])
            else:
                board.push(other.runAB(board))
            end = board.outcome()

            if len(board.move_stack) > 300:
                too_long = True
    except Exception as e:
        print(e)
        raise

    if too_long:
        return -1
    print(end)

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
        win = play_game_AB(model, True)
        if win < 0:
            win = 0.5
            aborted += 1
        score_w += win
        win = play_game_AB(model, False)
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