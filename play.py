import chess
import numpy as np
from src.bot import Bot
from src.canvas import Canvas
from src.robot import Robot
from src.vision import Vision


def is_game_over(board):
    return board.is_game_over() or board.can_claim_draw()

def is_move_legal(board_before, board_after):
    for move in board_before.legal_moves:
        board_before.push(move)
        matches = board_before.board_fen() == board_after.board_fen()
        board_before.pop()
        if matches:
            return True
    return False


if __name__ == "__main__":
    board = chess.Board()
    board.turn = chess.BLACK

    bot = Bot()
    canvas = Canvas()
    robot = Robot()
    vision = Vision()

    try:
        flange_to_base = robot.move_to_board()
        state = vision.detect_state(board, flange_to_base)
        board = state.board
        canvas.display_board(board, state.frame)

        while True: # not is_game_over(board):
            # Human (black)
            while True:
                canvas.wait_for_space_press()
                flange_to_base = robot.move_to_board()
                state = vision.detect_state(board, flange_to_base)
                if is_move_legal(board, state.board):
                    break
                canvas.display_board(state.board, state.frame, illegal=True)
            canvas.display_board(board, state.frame)

            # Bot (white)
            move = bot.get_move(state.board)
            robot.make_move(state, move)
            board = state.board
            board.push(move)
            canvas.display_board(board, state.frame)

        canvas.wait_for_q_press()

    finally:
        canvas.quit()
        vision.quit()
        bot.quit()
        robot.quit()