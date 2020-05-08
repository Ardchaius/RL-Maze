import numpy as np


class maze_environment:
    board = 0
    state = np.zeros((25, 1))

    def __init__(self):
        self.board = -np.ones((5, 5))  # The board also reflects reward for different actions
        self.board[2, 2] = -10  # Create a "hole" in the center space to see if the agent learns to avoid it
        self.board[1, 3] = -10
        self.board[3, 1] = -10
        self.board[4, 4] = 5
        self.state[0] = 1

    def make_move(self, move):
        # Position is a list of size 2 indicating row and column position in the board array
        # Move is an integer, 0 meaning up, 1 meaning right, 2 meaning down, and 3 meaning left
        lin_pos = np.argmax(self.state)
        row_pos = lin_pos // 5
        col_pos = lin_pos % 5
        if move == 0 and row_pos - 1 >= 0:
            row_pos -= 1
        elif move == 1 and col_pos + 1 < 5:
            col_pos += 1
        elif move == 2 and row_pos + 1 < 5:
            row_pos += 1
        elif move == 3 and col_pos - 1 >= 0:
            col_pos -= 1
        new_lin_pos = row_pos * 5 + col_pos
        if new_lin_pos == 24:
            terminal = 1
        else:
            terminal = 0

        reward = self.board[row_pos, col_pos]
        self.state[lin_pos] = 0
        self.state[new_lin_pos] = 1

        return terminal, reward

    def reset(self):
        self.state = np.zeros((25, 1))
        self.state[0] = 1
