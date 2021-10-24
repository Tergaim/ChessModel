import fnmatch
import os
import numpy as np
import chess.pgn
from tqdm import tqdm


def find_files(directory, pattern='*.pgn'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_text(directory):
    '''Generator that yields text raw from the directory.'''
    files = [directory]
    labels = []
    for filename in files:
        k = 0
        l = 0
        white = 0
        black = 0
        pgn = open(filename)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break 
            
            node = game
            move_list = []
            win = 1 if game.headers["Result"] == "1-0" else 0
            save = ("/2" not in game.headers["Result"]) and ((win == 1 and white < 10000) or (win == 0 and black < 10000))
            move_list.append("1" if game.headers["Result"] == "1-0" else "0")
            while not node.is_end():
                next_node = node.variation(0)
                label = next_node.move.uci()
                move_list.append(label)
                node = next_node
            if save and node.board().outcome() is not None:
                labels.append(f"{' '.join(move_list)}\n")
                if win == 1:
                    white += 1
                else:
                    black += 1
                l += 1
            if k % 100 == 0 and k > 1:
                print (f"game: {k}\tselected: {str(white).zfill(5)} white and {str(black).zfill(5)} black")
            k += 1
            if l == 20000:
                break
        pgn.close()
        
    y = np.array(labels)
    print(y.shape)
    np.savetxt("moves_data_202001.txt", y, delimiter="", newline="", fmt="%s")


def main():
    load_generic_text("/media/tergaim/Data/chess_data/Lichess_Elite_Database/pgn/lichess_elite_2020-01.pgn")
    print ("Done.")

if __name__ == '__main__':
    main()