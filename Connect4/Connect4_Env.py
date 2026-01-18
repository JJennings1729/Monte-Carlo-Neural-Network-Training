import numpy as np
from numba import njit

# Game constants
N_PLAYERS = 2
TOT_ACTIONS = 7
ROWS, COLS = 6, 7
BOARD_LEN = ROWS * COLS
MAX_VAL = 2
MAX_DEPTH = 43
START_BOARD = np.zeros(BOARD_LEN, np.int32)
FULL_BOARD_SHAPE = (2, ROWS, COLS)

# Network path
model_path = "Connect4/Connect4_Network.pth"


# Zobrist basic arrays
ZB = np.random.randint(0, 2**64, size=(BOARD_LEN, MAX_VAL), dtype=np.uint64)
ZP = np.random.randint(0, 2**64, size=(N_PLAYERS,), dtype=np.uint64)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

@njit
def idx(row, col):  # Quick convert from row + col to flat idx
    return row * COLS + col

@njit
def drop_piece(board, player, action, out_board):
    # manually copy board
    for i in range(BOARD_LEN):
        out_board[i] = board[i]

    # drop the piece in the column
    for r in range(ROWS-1, -1, -1):
        i = idx(r, action)
        if out_board[i] == 0:
            out_board[i] = player
            return r, action

    return -1, -1  # column full

@njit
def check_winner(board, row, col, player):
    directions = [(1,0), (0,1), (1,1), (1,-1)]  # vertical, horizontal, diag1, diag2

    for dr, dc in directions:
        count = 1
        # forward direction
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[idx(r, c)] == player:
            count += 1
            r += dr
            c += dc
        # backward direction
        r, c = row - dr, col - dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[idx(r, c)] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 4:
            return True
    return False

@njit
def step(board, player, done, winner, action, out_board):
    if done or action < 0:
        for i in range(BOARD_LEN):
            out_board[i] = board[i]
        return player, done, winner

    row, col = drop_piece(board, player, action, out_board)

    done = False
    winner = 0

    if check_winner(out_board, row, col, player):
        done = True
        winner = player
    else:
        # check draw
        full = True
        for c in range(COLS):
            if out_board[idx(0, c)] == 0:
                full = False
                break
        if full:
            done = True
            winner = 0

    next_player = 1 if player == 2 else 2
    return next_player, done, winner

@njit
def legal_actions(board, player, out_actions):
    cnt = 0
    for c in range(COLS):
        if board[idx(0,c)] == 0:
            out_actions[cnt] = c
            cnt += 1
    return cnt

@njit
def full_board(board, player, out_arr):
    opp = (player % N_PLAYERS) + 1
    for r in range(ROWS):
        for c in range(COLS):
            val = board[idx(r,c)]
            out_arr[0, r, c] = 1.0 if val == player else 0.0
            out_arr[1, r, c] = 1.0 if val == opp else 0.0

@njit
def make_hash(board, player):
    h = np.uint64(0)
    for b in range(BOARD_LEN):
        v = board[b]
        h ^= ZB[b, v]
    h ^= ZP[player-1]
    return h

@njit
def action_mask(action, board, player):
    # find row to place
    R = -1
    for r in range(ROWS-1, -1, -1):
        if board[r * COLS + action] == 0:
            R = r
            break
    if R == -1:
        return np.uint64(0)
    
    old_val = board[R * COLS + action]
    new_val = player if old_val == 0 else 0

    mask = ZB[R * COLS + action, old_val] ^ ZB[R * COLS + action, new_val]
    next_player = 1 if player == 2 else 2
    mask ^= ZP[player-1] ^ ZP[next_player-1]
    return np.uint64(mask)

@njit
def random_games(
    boards, players,
    tmp_board, tmp_actions,
    START_BOARD, max_moves
):
    """
    Fill `boards` and `players` with random non-terminal positions.

    Constraints:
    - No slicing on RHS
    - No temporary array allocation
    - step() must NOT mutate input board
    - step() writes next state into tmp_board
    """

    B = boards.shape[0]
    BOARD_LEN = boards.shape[1]

    for idx in range(B):

        # Initialize board in-place (NO slice copy)
        for i in range(BOARD_LEN):
            boards[idx, i] = START_BOARD[i]

        players[idx] = 1
        done = False
        winner = 0

        n_moves = np.random.randint(1, max_moves + 1)

        for _ in range(n_moves):

            n_legal = legal_actions(boards[idx], players[idx], tmp_actions)
            if n_legal == 0:
                # Keep current position (initial if no moves made)
                break

            move = tmp_actions[np.random.randint(n_legal)]

            new_player, new_done, new_winner = step(
                boards[idx], players[idx],
                done, winner,
                move,
                tmp_board
            )

            # Stop on terminal positions (ignore them)
            if new_done:
                break

            # Copy tmp_board → boards[idx] elementwise (NO allocation)
            for j in range(BOARD_LEN):
                boards[idx, j] = tmp_board[j]

            players[idx] = new_player
            done = new_done
            winner = new_winner