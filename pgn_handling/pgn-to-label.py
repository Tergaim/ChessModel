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
    files = find_files(directory)
    labels = set()
    for filename in files:
        k = 0
        pgn = open(filename)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break 
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                label = next_node.move.uci()
                labels.add(label)
                node = next_node
            if k % 100 == 0 and k > 1:
                print ("Labeling file: " + filename + ", game: " + str(k))
            k += 1
        pgn.close()
        
    y = np.array(list(labels))
    print(y.shape)
    np.savetxt("labels.txt", y, delimiter=" ", newline=" ", fmt="%s")


def main():
    load_generic_text("/media/tergaim/Data/chess_data/Lichess Elite Database/pgn/")
    print ("Done.")

if __name__ == '__main__':
    main()