# Chess engine using Supervised and Reinforcement Learning

This project was done as an IDP at TUM Informatics.

The model is trained first by supervised learning to match human moves, and then can be further enhanced by playing against itself.
Furthermore, while the model can be used without tree search, it is stronger with it.
Two strategies were implemented in this project:
 - probability search: similar to an alpha-beta search, but the network is used to compute an existence probability for each node. The search is stopped when the probability drops below a user-defined threshold.
 - MCTS: from a given position, games are played until completion starting with every possible move. Each move is graded according to how many games were won using it, and the move leading to the highest number of victories is chosen.

The code for these two searches can be found in `compet_MCTS.py` and `compet_Prob.py`.
Using MCTS is the recommended one: it is way faster than probability search, with a difference of ~180 Elo in favor of MCTS.

## Structure

The code for formatting pgn data can be found in `./pgn_handling`.
 - `pgn-to-txt.py` is used to format data for supervised learning
 - `pgn-to-move-list.py` extracts 20,000 games (one half won by white, the other by black). Can be used to enhance the self-play.
 - `pgn-to-label.py` extracts moves present in your dataset and replaces `./labels/labels.txt`. The provided one contains all moves present in Lichess Elite Database in UCI formatting (4 or 5 letters describing start square - destination square - promotion).
 - `train_supervised.py` and `train_reinforce.py` are used to train the model.
 - `weights.py` contains parameters for the evaluation function used in Alpha-Beta and Probability search.
