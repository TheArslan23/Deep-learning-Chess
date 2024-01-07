# Deep-learning-Chess
I love chess, and after learning some fundamentals of deep learning, I came across a very interesting paper DeepChess which tends to give a competition to a Human Chess Grandmaster. I decided to give its implementation a try using Tensorflow.

To play: Install python-chess, and then from the main directory, run: python game.py

To train: This model was trained with:

CUDA 7.5
Tensorflow 0.10.0
Run python train.py to train the model on the data available in the folder 'pGames' Some older network checkpoints can be found in the folder 'net'.

To mine a different dataset: Run python get_data.py, but be sure to change the file name in the source code.

Some notes: The basic idea of the paper is that we can get a deep network to play chess by teaching it an evaluation function that takes in 2 positions and outputs the better one. The network can then be used in a modified Alpha-Beta pruning algorithm, where instead of comparing between two positions' evaluations (as numbers), we compare between the positions themselves.
