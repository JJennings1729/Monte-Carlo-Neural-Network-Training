import py5
import numpy as np
from Connect4.Connect4_Env import *

def Connect4PlayGame(AI_player, human_player=1):

    background_col = py5.color("#f542e6")
    board_col = py5.color("#415afa")
    player_cols = [py5.color("#e33217"), py5.color("#f5f11d")]


    state = {
        "board": START_BOARD, 
        "player": 1,
        "done": False,
        "winner": 0,
        "legal_indices": None,
        "legal_moves": np.zeros(TOT_ACTIONS, dtype=np.int32),
        "AI player": AI_player,
        "Human player": human_player
    }


    new_counter = {
        "x": 0.0,
        "y": 0.0,
        "dest": 0.0,
        "color": player_cols[0],
        "speed": 0.0,
        "moving": False,
        "accel": 1,
        "bounce": 0.6
    }


    hole_radius = 35
    hole_gap = 20
    padding_x = 50
    padding_y = 20
    screen_x, screen_y = 800, 700

    spacing_x = hole_radius * 2 + hole_gap
    spacing_y = hole_radius * 2 + hole_gap
    board_w = spacing_x * COLS + 2 * padding_x
    board_h = spacing_y * ROWS + 2 * padding_y
    diff_x = (screen_x - board_w) // 2
    diff_y = (screen_y - board_h) // 2

    def settings():
        py5.size(screen_x, screen_y)

    def setup():
        py5.no_stroke()
        global board_graphics

        board_graphics = draw_board(
            board_w, board_h,
            ROWS, COLS,
            hole_radius,
            spacing_x, spacing_y,
            padding_x, padding_y
        )

        state["legal_indices"] = legal_actions(
            state["board"],
            state["player"],
            state["legal_moves"]
        )

        py5.background(background_col)
        py5.image(board_graphics, diff_x, diff_y)
        for row in range(ROWS):
            for col in range(COLS):
                val = state["board"][row * ROWS + col]
                if val > 0:
                    color = val - 1
                    cx = padding_x + col * spacing_x + spacing_x // 2 + diff_x
                    cy = padding_y + row * spacing_y + spacing_y // 2 + diff_y
                    py5.fill(player_cols[color])
                    py5.ellipse(cx, cy, 2*hole_radius, 2*hole_radius)

    def draw():
        if not state["done"] and not new_counter["moving"]:
            if state["player"] != state["Human player"]:
                AI_policy = state["AI player"](state["board"], state["player"])
                if sum(AI_policy) == 0.0:
                    AI_move = -1    # Skip move
                else:
                    legals = state["legal_moves"][:state["legal_indices"]]
                    AI_move = np.random.choice(len(AI_policy), p=AI_policy)
                    if AI_move not in legals:
                        print("ERROR: Selected move by AI is illegal")
                make_move(AI_move)
        
        elif new_counter["moving"]:
            py5.fill(background_col)
            py5.ellipse(new_counter["x"], new_counter["y"], 2*hole_radius+10, 2*hole_radius+10)
            new_counter["speed"] += new_counter["accel"]
            new_counter["y"] += new_counter["speed"]
            if abs(new_counter["y"] - new_counter["dest"]) < new_counter["speed"] + new_counter["accel"]:
                if abs(new_counter["speed"]) < 5:
                    new_counter["y"] = new_counter["dest"]
                    new_counter["moving"] = False
                else: 
                    new_counter["speed"] *= -new_counter["bounce"]
            py5.fill(new_counter["color"])
            py5.ellipse(new_counter["x"], new_counter["y"], 2*hole_radius, 2*hole_radius)

        py5.image(board_graphics, diff_x, diff_y)


    def draw_board(w, h, rows, cols, rad, spacing_x, spacing_y, pad_x, pad_y):
        # Create an empty PImage instead of a Graphics object
        board_img = py5.create_image(w, h, py5.ARGB)
        board_img.load_pixels()

        # Fill background with board color
        for i in range(w * h):
            board_img.pixels[i] = board_col

        # Draw the holes by setting alpha to 0
        for r in range(rows):
            for c in range(cols):
                cx = pad_x + c * spacing_x + spacing_x // 2
                cy = pad_y + r * spacing_y + spacing_y // 2

                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        if dx * dx + dy * dy <= rad * rad:
                            px = cx + dx
                            py = cy + dy
                            if 0 <= px < w and 0 <= py < h:
                                idx = py * w + px
                                board_img.pixels[idx] = py5.color(0, 0, 0, 0)  # transparent hole

        board_img.update_pixels()
        return board_img


    def mouse_clicked():  # Human move
        if not state["done"] and not new_counter["moving"]:
            if state["player"] != state["Human player"]:
                return None
            if not diff_x + padding_x < py5.mouse_x < diff_x + board_w - padding_x:
                return None
            if not diff_y + padding_y < py5.mouse_y < diff_y + board_h - padding_y:
                return None
            action = (py5.mouse_x - (diff_x + padding_x)) // spacing_x
            legals = state["legal_moves"][:state["legal_indices"]]
            if action in legals:
                make_move(action)
            else:
                print(f"Cannot play to Column {action}")

    def make_move(action):

        new_board = np.zeros_like(state["board"])

        row, _ = drop_piece(state["board"], state["player"], action, new_board)
        
        new_counter["x"] = diff_x + padding_x + spacing_x * (action + 0.5)
        new_counter["y"] = -50
        new_counter["dest"] = diff_y + padding_y + spacing_y * (row + 0.5)
        new_counter["color"] = player_cols[state["player"]-1]
        new_counter["speed"] = 0.0
        new_counter["moving"] = True

        state["player"], state["done"], state["winner"] = \
            step(state["board"], state["player"], state["done"], state["winner"], action, new_board)
        state["board"] = new_board.copy()
        state["legal_indices"] = legal_actions(
            state["board"],
            state["player"],
            state["legal_moves"]
        )
        if state["done"]:
            print("Game completed!")
            match state['winner']:
                case 0: print("Game is drawn")
                case 1: print("Red player has won!")
                case 2: print("Yellow player has won!")

    py5.run_sketch()