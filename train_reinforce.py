"""Training script for the network."""

import fnmatch
import os
import random

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from chess import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                   Board, Move, Outcome, Termination)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ChessModel
from helpers import to_planes, choose_move

# print(torch.__version__)
# raise KeyboardInterrupt

DATA_FILE = './moves_data_202001.txt'
LABELS_FILE = './labels/labels.txt'
save_path = '/media/tergaim/Data/chess_data/experiment_rl/train_with_endgame_vfinal4.pth'
base_path = '/media/tergaim/Data/chess_data/experiment_rl/base.pth'

BATCH_SIZE = 100
FEATURE_PLANES = 13
LABEL_SIZE = 1965
EPOCHS = 4

available_labels = []
# missing labels :  e2d1b c2b1b
label_to_idx = {}
writer = SummaryWriter(log_dir="runs/train_with_endgame_vfinal4")


def read_labels():
    labels_array = []
    with open(LABELS_FILE) as f:
        lines = str(f.readlines()[0]).split(" ")
        for label in lines:
            if(label != " " and label != '\n'):
                labels_array.append(label)
    print(f"label size = {len(labels_array)}")
    return labels_array


def read_data():
    data = []
    white = 0
    black = 0
    with open(DATA_FILE) as f:
        lines = f.readlines()
        for game in lines:
            string = game.replace('\n', '')
            move_list = string.split(' ')
            data.append(move_list)
            if move_list[0] == '1':
                white += 1
            else:
                black += 1
    print(f"dataset size = {len(data)}")
    print(white)
    print(black)
    return data


def imitate_game(model, move_list, white_wins):
    board = Board()
    white_player = [np.full((8, 8), 1, dtype=int)]
    black_player = [np.full((8, 8), 0, dtype=int)]

    board = Board()
    planes_0W = to_planes(board)
    planes_1W = planes_0W
    planes_2W = planes_0W
    planes_0B = planes_0W
    planes_1B = planes_0W
    planes_2B = planes_0W

    prediction_stack_white = []
    played_white = []
    available_moves_white = []
    prediction_stack_black = []
    played_black = []
    available_moves_black = []

    too_long = False
    end = None
    white = True
    counter = 0
    try:
        start_rl = random.randint(max(0, len(move_list)-15), len(move_list)-2)
    except ValueError:
        start_rl = len(move_list)-5
    for move in move_list[:start_rl]:
        extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
        if white:
            planes_0W = planes_1W
            planes_1W = planes_2W
            planes_2W = to_planes(board)
            # if white_wins == 1:
            #     preds, move_idx, available_moves = choose_move(model, planes_0W, planes_1W, planes_2W, white_player, extra, board, label_to_idx)
            #     prediction_stack_white.append(preds)
            #     played_white.append(move_idx)
            #     available_moves_white.append(available_moves)
        else:
            planes_0B = planes_1B
            planes_1B = planes_2B
            planes_2B = to_planes(board)
            # if white_wins == 0:
            #     preds_b, move_idx, available_moves = choose_move(model, planes_0B, planes_1B, planes_2B, black_player, extra, board, label_to_idx)
            #     prediction_stack_black.append(preds_b)
            #     played_black.append(move_idx)
            #     available_moves_black.append(available_moves)
        board.push_uci(move)
        white = not white

    while counter < start_rl*10 and end is None:
        extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]
        try:
            if white:
                planes_0W = planes_1W
                planes_1W = planes_2W
                planes_2W = to_planes(board)
                preds, move_idx, available_moves = choose_move(model, planes_0W, planes_1W, planes_2W, white_player, extra, board, label_to_idx)
                prediction_stack_white.append(preds)
                played_white.append(move_idx)
                available_moves_white.append(available_moves)

            else:
                planes_0B = planes_1B
                planes_1B = planes_2B
                planes_2B = to_planes(board)
                preds_b, move_idx, available_moves = choose_move(model, planes_0B, planes_1B, planes_2B, black_player, extra, board, label_to_idx)
                prediction_stack_black.append(preds_b)
                played_black.append(move_idx)
                available_moves_black.append(available_moves)
            board.push_uci(available_labels[move_idx])

        except Exception as e:  # happens when no legal move is possible but wasn't detected by board.outcome()
            counter = 1000  # end the game in a draw
            # print(e)

        counter += 1
        white = not white
        end = board.outcome()

    draw = False
    aborted = False
    win = 0.5
    if end is None:
        aborted = True
    else:
        win = 1 if end.winner else 0
        if end.termination != Termination.CHECKMATE:
            draw = True

    return prediction_stack_white, played_white, available_moves_white, prediction_stack_black, played_black, available_moves_black, draw, aborted, win, start_rl


def accuracy(predictions, labels):
    # print('start')
    # for i in range(min(BATCH_SIZE,5)):
    #     print(f'{available_labels[torch.argmax(predictions[i])]} == {available_labels[torch.argmax(labels[i])]}')
    return torch.sum(torch.argmax(predictions, 1) == torch.argmax(labels, 1)) / float(predictions.shape[0])


def check(p1, p2, labels, boards):
    print('start')
    for i in range(BATCH_SIZE):
        if torch.argmax(p1[i]) == torch.argmax(labels[i]) and torch.argmax(p1[i]) != torch.argmax(p2[i]):
            print(boards[i])
            print(f'{available_labels[torch.argmax(p1[i])]} != {available_labels[torch.argmax(p2[i])]}, label {available_labels[torch.argmax(labels[i])]}')


def rescale(x, mask):
    x = x*mask
    x = x/(torch.sum(x, dim=1).unsqueeze(1) + 0.0001)
    return x


def main():
    torch.cuda.set_device(0)
    global available_labels
    global label_to_idx
    available_labels = read_labels()
    for i, l in enumerate(available_labels):
        label_to_idx[l] = i
    data = read_data()
    model = ChessModel(3*FEATURE_PLANES+2).cuda()
    model_ref = ChessModel(3*FEATURE_PLANES+2).cuda()

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    print(params.shape)

    # return

    try:
        model.load_state_dict(torch.load(save_path), strict=False)
        model_ref.load_state_dict(torch.load(save_path), strict=False)
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    loss_func = nn.BCELoss()
    params = list(model.classify1.parameters()) + list(model.s8.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Training...')
    step = 1
    loss_aggreg = 0
    for epoch in tqdm(range(EPOCHS)):
        for movelist in tqdm(data):
            # if step < 52001:
            #     step +=1
            #     continue
            supposed_winner = 1 if movelist[0] == "1" else 0
            prediction1, played1, available_moves1, prediction2, played2, available_moves2, draw, aborted, win, start_rl = imitate_game(model, movelist[1:], supposed_winner)

            if len(prediction1) == 0 or len(prediction2) == 0:
                step += 1
                continue

            optimizer.zero_grad()
            predictions = None
            labels = None

            predictions1 = torch.stack(prediction1)
            labels1 = predictions1.clone().detach()
            for i in range(len(played1)):
                try:
                    # if i < start_rl and supposed_winner == 1:
                    #     labels1[i][played1[i]] = 1
                    # else:
                    if win == 1 or (draw and supposed_winner == 0):
                        labels1[i][played1[i]] = 1
                    elif aborted or (draw and supposed_winner == 1):
                        labels1[i][played1[i]] = 1/available_moves1[i]
                    elif win == 0 and supposed_winner == 1:
                        labels1[i][played1[i]] = 0
                except:
                    pass

            predictions2 = torch.stack(prediction2)
            labels2 = predictions2.clone().detach()
            for i in range(len(played2)):
                try:
                    # if i < start_rl and supposed_winner == 0:
                    #     labels2[i][played2[i]] = 1
                    # else:
                    if win == 0 or (draw and supposed_winner == 1):
                        labels2[i][played2[i]] = 1
                    elif aborted or (draw and supposed_winner == 0):
                        labels2[i][played2[i]] = 1/available_moves2[i]
                    elif win == 1 and supposed_winner == 0:
                        labels2[i][played2[i]] = 0
                except:
                    pass

            loss = loss_func(torch.cat([predictions1, predictions2]), torch.cat([labels1, labels2]))
            # if win != 0.5 and win != supposed_winner:
            #     loss *= 2
            loss.backward()
            loss_aggreg += loss.item()
            optimizer.step()

            # training logs
            if (step % 10 == 0):
                writer.add_scalar('Loss/train', loss_aggreg, step)
                loss_aggreg = 0
            if (step % 2000 == 0):
                torch.save(model.state_dict(), save_path)
                score = test(model, model_ref, step)
                if score > 50:
                    print(f"The model achieved a score of {score}. Replacing reference.")
                    model_ref.load_state_dict(torch.load(save_path))

            step += 1
    # save progress at the end
    torch.save(model.state_dict(), save_path)


def play_game_test(model, model_ref):
    w = model
    b = model_ref

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
            planes_0W = planes_1W
            planes_1W = planes_2W
            planes_2W = to_planes(board)
            extra = [np.full((8, 8), board.halfmove_clock, dtype=int)]

            _, move_idx, _ = choose_move(w, planes_0W, planes_1W, planes_2W, white_player, extra, board, label_to_idx)

            board.push_uci(available_labels[move_idx])
            end = board.outcome()
            if end is not None:
                break

            planes_0B = planes_1B
            planes_1B = planes_2B
            planes_2B = to_planes(board)

            _, move_idx, _ = choose_move(b, planes_0B, planes_1B, planes_2B, white_player, extra, board, label_to_idx)

            board.push_uci(available_labels[move_idx])
            end = board.outcome()

            if len(board.move_stack) > 300:
                too_long = True
    except Exception as e:
        print(e)
        raise

    # if end is not None:
    #     print(end)
    # else:
    #     print('Max move reached, aborted.')

    if too_long:
        return -1

    win = 0.5
    if end.winner == WHITE:
        win = 1
    elif end.winner == BLACK:
        win = 0

    return win


def test(model, model_ref, epoch):
    score_w = 0.0
    score_b = 0.0
    aborted = 0
    draw = 0
    for n in tqdm(range(200)):
        win = play_game_test(model, model_ref)
        if win < 0:
            win = 0.5
            aborted += 1
        score_w += win
        win = play_game_test(model_ref, model)
        if win == 0.5:
            draw += 1
        if win < 0:
            win = 0.5
            aborted += 1
        score_b += 1-win
    # print(f'score = {score}')
    writer.add_scalar('Test/Score_W', score_w/2, epoch)
    writer.add_scalar('Test/Score_B', score_b/2, epoch)
    writer.add_scalar('Test/Score_All', (score_w + score_b)/4, epoch)
    writer.add_scalar('Test/Draws', draw, epoch)
    writer.add_scalar('Test/Aborted', aborted, epoch)
    return (score_w + score_b)/4


if __name__ == '__main__':
    main()
