"""Training script for the network."""

import os
import fnmatch
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from model import ChessModel
from tqdm import tqdm
from chess import Move, Board
from torch.utils.tensorboard import SummaryWriter
from helpers import find_files

TRAIN_DIRECTORY = ''
VALIDATION_DIRECTORY = ''
LABELS_DIRECTORY = './labels'
SAVE_PATH = ''

BATCH_SIZE = 200
FEATURE_PLANES = 13
LABEL_SIZE = 1965
EPOCHS = 20

available_labels = []
writer = SummaryWriter(log_dir="runs")


def _read_text(filename, batch_size):
    with open(filename) as f:
        text = f.readlines()[:-1]
        games = []
        move_number = 0
        for _ in range(batch_size):
            while move_number < 10:
                i = random.randint(0, len(text)-1)
                game = text[i]
                move_number = int(game.split(':')[0].split(" ")[5])
            games.append((text[i-4], text[i-2], text[i]))
            move_number = 0
        return games


def generate_batch(batch_size, directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    filename = random.choice(files)
    text = _read_text(filename, batch_size)
    return text


def read_labels(directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    labels_array = []
    for filename in files:
        with open(filename) as f:
            lines = str(f.readlines()[0]).split(" ")
            for label in lines:
                if(label != " " and label != '\n'):
                    labels_array.append(label)
    print(f"label size = {len(labels_array)}")
    return labels_array


value = {'p': 1, 'P': 1, 'r': 5, 'R': 5, 'n': 3, 'N': 3, 'b': 3, 'B': 3, 'q': 10, 'Q': 10, 'k': 100, 'K': 100, '1': 0}


def to_planes(board_pieces: list):
    # # All pieces plane
    # board_all = [value[val] for val in board_pieces]
    # board_all = np.reshape(board_all, (8, 8))
    # Only spaces plane
    board_blank = [int(val == '1') for val in board_pieces]
    board_blank = np.reshape(board_blank, (8, 8))
    # # Only white plane
    # board_white = [int(val.isupper()) for val in board_pieces]
    # board_white = np.reshape(board_white, (8, 8))
    # # Only black plane
    # board_black = [int(not val.isupper() and val != '1') for val in board_pieces]
    # board_black = np.reshape(board_black, (8, 8))

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

    planes = np.stack((
        # np.copy(board_all),
        np.copy(board_blank),
        # np.copy(board_black),
        # np.copy(board_white),
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


def reformat(datas, labels):
    for game0, game1, game2 in datas:
        try:
            board_state1 = (game1.split(":")[0]).replace("/", "")
            board_state0 = (game0.split(":")[0]).replace("/", "")
            game2fen = game2.split(":")[0]
            board_state2 = game2fen.replace("/", "")
            game2fen = game2fen.replace("11111111", "8")
            game2fen = game2fen.replace("1111111", "7")
            game2fen = game2fen.replace("111111", "6")
            game2fen = game2fen.replace("11111", "5")
            game2fen = game2fen.replace("1111", "4")
            game2fen = game2fen.replace("111", "3")
            game2fen = game2fen.replace("11", "2")
            board = Board(game2fen)
            label_state = board.parse_san(game2.split(":")[1].replace("\n", "")).uci()

            legal_moves = []
            for move in board.legal_moves:
                legal_moves.append(board.uci(move))
            # print(board)
            # print(label_state)
        except Exception as e:
            print(e)
            continue
        label = np.zeros(LABEL_SIZE)
        for i in range(LABEL_SIZE):
            if(label_state == labels[i]):
                label[i] = 1

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

        planes = np.concatenate((game_feat0,
                                 game_feat1,
                                 game_feat2,
                                 [current_player],
                                 [extra]),
                                axis=0
                                )
        planes = np.reshape(planes, (3*FEATURE_PLANES+2, 8, 8))
        yield (planes, label, legal_moves, board.fullmove_number)


def accuracy(predictions, labels):
    return torch.sum(torch.argmax(predictions, 1) == torch.argmax(labels, 1)) / float(predictions.shape[0])


def test_number(predictions, labels):
    return torch.argmax(predictions, 1) == torch.argmax(labels, 1)


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
    files = find_files(TRAIN_DIRECTORY, "*.txt")
    val_files = find_files(VALIDATION_DIRECTORY, "*.txt")
    torch.cuda.set_device(0)
    global available_labels
    available_labels = read_labels(LABELS_DIRECTORY, "*.txt")
    model = ChessModel(3*FEATURE_PLANES+2).cuda()

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    print(params.shape)

    try:
        model.load_state_dict(torch.load(SAVE_PATH))
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Training...')
    step = 0

    loss_agreg = 0
    for epoch in tqdm(range(EPOCHS)):
        for filename in tqdm(files):
            step += 1
            # if step < 3510*15:
            #     continue
            # prepare batch
            train_batch = _read_text(filename, BATCH_SIZE)
            train_dataset = reformat(train_batch, available_labels)

            batch_data = []
            batch_labels = []
            batch_legal_moves = []
            for (plane, label, legal_moves, _) in train_dataset:
                batch_data.append(plane)
                batch_labels.append(label)

                legal_labels = [int(mov in legal_moves) for mov in available_labels]
                batch_legal_moves.append(legal_labels)
            batch_labels = torch.FloatTensor(batch_labels).cuda()

            # compute
            optimizer.zero_grad()
            output_train = model(torch.FloatTensor(batch_data).cuda())
            mask_legal = torch.FloatTensor(batch_legal_moves).cuda()
            # if epoch > 2:
            #   loss = loss_func(rescale(output_train, mask_legal), batch_labels)
            # else:
            loss = loss_func(output_train, batch_labels)
            loss.backward()
            loss_agreg += loss.item()
            optimizer.step()

            # training logs
            if (step % 100 == 0):
                writer.add_scalar('Loss/train', loss_agreg, step)
                loss_agreg = 0
                acc_train = accuracy(output_train, batch_labels)
                writer.add_scalar('Acc_all/train', acc_train, step)
                output_train2 = output_train * torch.FloatTensor(batch_legal_moves).cuda()
                acc_leg = accuracy(output_train2, batch_labels)
                writer.add_scalar('Acc_leg/train', acc_leg, step)
            if (step % 1000 == 0):
                test(val_files, loss_func, model, step)
                torch.save(model.state_dict(), SAVE_PATH)

    # save progress at the end
    test(val_files, loss_func, model, step)
    torch.save(model.state_dict(), SAVE_PATH)


def test(val_files, loss_func, model, step):
    # writer.add_histogram('conv1.bias', model.first.bias, 0)
    # writer.add_histogram('conv1.weight', model.first.weight, 0)
    # writer.add_histogram('conv1.weight.grad', model.first.weight.grad, 0)

    real = [0 for i in range(10)]
    found = [0 for i in range(10)]

    board = Board()

    loss_val = 0
    acc_all_val = 0
    acc_leg_val = 0
    for filename in tqdm(val_files):
        # prepare val batch; compute boards for leg_all
        validation_batch = generate_batch(BATCH_SIZE, VALIDATION_DIRECTORY, "*.txt")
        validation_dataset = reformat(validation_batch, available_labels)
        batch_valid_data = []
        batch_valid_labels = []
        batch_legal_moves = []
        boards_move = []
        for (plane, label, legal_moves, number) in validation_dataset:
            # print(f"board len : {len(board)}")
            batch_valid_data.append(plane)
            batch_valid_labels.append(label)
            legal_labels = [int(mov in legal_moves) for mov in available_labels]
            batch_legal_moves.append(legal_labels)
            if number > 90:
                number = 9
            else:
                number = int((number+1)/10)
            boards_move.append(number)
            real[number] += 1

        # compute
        mask_legal = torch.FloatTensor(batch_legal_moves).cuda()
        output_val = model(torch.FloatTensor(batch_valid_data).cuda())
        batch_valid_labels = torch.FloatTensor(batch_valid_labels).cuda()
        acc_all = accuracy(output_val, batch_valid_labels)
        output_val2 = rescale(output_val, mask_legal)
        acc_leg = accuracy(output_val2, batch_valid_labels)
        result = test_number(output_val2, batch_valid_labels).cpu().detach().numpy()
        for i, r in enumerate(result):
            if r:
                found[boards_move[i]] += 1
        loss = loss_func(output_val2, batch_valid_labels)
        loss_val += loss.item()
        acc_all_val += acc_all.item()
        acc_leg_val += acc_leg.item()

    # log
    writer.add_scalar('Loss/val', loss_val/len(val_files), step)
    writer.add_scalar('Acc_all/val', acc_all_val/len(val_files), step)
    writer.add_scalar('Acc_leg/val', acc_leg_val/len(val_files), step)

    # for i in range(10):
    #     writer.add_scalar('Test/Hist', found[i]/real[i], 5+10*i)
    #     writer.add_scalar('Test/Values', real[i], 5+10*i)
    #     writer.add_scalar('Test/Found', found[i], 5+10*i)
    # writer.close()


if __name__ == '__main__':
    main()
