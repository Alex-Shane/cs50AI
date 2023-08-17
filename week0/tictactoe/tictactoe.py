"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state():
        return X
    elif terminal(board):
        return None
    else:
        x = 0
        o = 0
        for k in range(3):
            for i in range(3):
                if board[k][i] == X:
                    x += 1
                elif board[k][i] == O:
                    o += 1
                else:
                    pass
        if x > o:
            return O
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return None
    else:
        actions = []
        for x in range(3):
            for i in range(3):
                if board[x][i] == EMPTY:
                    actions.append((x,i))
        return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board_copy = []
    for x in range(3):
        board_copy.append([EMPTY,EMPTY,EMPTY])
        for i in range(3):
            board_copy[x][i] = board[x][i]
    row = action[0]
    col = action[1]
    if board_copy[row][col] != EMPTY:
        raise ValueError(f'Invalid action at ({row},{col}) with current board')
    else:
        turn = player(board_copy)
        board_copy[row][col] = turn
        return board_copy
    


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for x in range(3):
        if board[x] == [X,X,X]:
            return X
        if board[x] == [O,O,O]:
            return O
    for x in range(3):
        col = []
        for i in range(3):
            col.append(board[i][x])
        if col == [X,X,X]:
            return X
        if col == [O,O,O]:
            return O
    left_diagonal = []
    right_diagonal = []
    for x in range(3):
        for i in range(3):
            if x == i:
                left_diagonal.append(board[x][i])
            if i == 3-x-1:
                right_diagonal.append(board[x][i])
    if left_diagonal == [X,X,X]:
        return X
    elif left_diagonal == [O,O,O]:
        return O
    elif right_diagonal == [X,X,X]:
        return X
    elif right_diagonal == [O,O,O]:
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    empty_cells = sum(row.count(EMPTY) for row in board)
    if empty_cells == 0:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    else:
        turn = player(board)
        plays = actions(board)
        if turn == X:
            val = float('-inf')
            best = None
            alpha = float('-inf')
            for action in plays:
                cur_val = Min(result(board, action), alpha, float('inf'))
                if cur_val > val:
                    val = cur_val
                    best = action
                alpha = max(alpha, cur_val)
            return best
        else:
            val = float('inf')
            best = None
            beta = float('inf')
            for action in plays:
                cur_val = Max(result(board,action), float('-inf'), beta)
                if cur_val < val:
                    val = cur_val
                    best = action
            return best

def Min(board, alpha, beta):
    v = float('inf')
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = min(v, Max(result(board,action),alpha,beta))
        if v <= alpha:
            break
        beta = min(beta, v)
    return v

def Max(board, alpha, beta):
    v = float('-inf')
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = max(v, Min(result(board,action),alpha,beta))
        if v >= beta:
            break
        alpha = max(alpha,v)
    return v
    
    
    
    
    
    
    
    
    
