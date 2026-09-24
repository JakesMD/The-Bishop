import chess
import numpy as np
from src.robot import *
from src.chess_bot import *
from src.canvas import *
from src.motion_planner import *
from src.vision import *
from src.types import *
from src.board import *

chess_bot = ChessBot()
canvas = Canvas()
robot = Robot()
planner = MotionPlanner(robot)
vision = Vision()


def set_viewing_pose():
    while True:
        planner.set_viewing_pose()
        state = vision.detect_state(chess.Board())

        if state.board is not None:
            return state

        canvas.set_text("Line up the board",
                        "Move the arm or the playing surface until all four corner tags are in shot.")
        canvas.set_frame(state.frame)
        canvas.wait_for_space_press("look again")

def get_state(old_board, is_setup=False):
    while True:
        planner.look_at_board()
        state = vision.detect_state(old_board)

        if state.board is None:
            canvas.set_text("The board isn't visible",
                            "Put the playing surface back fully below the camera.")
            canvas.set_frame(state.frame)
            canvas.wait_for_space_press("look again")
        elif not is_setup and not is_move_legal(old_board, state.board):
            canvas.set_text("That isn't a legal move",
                            "Put the pieces back the way they were and play again.")
            canvas.set_detected_board(state.board)
            canvas.set_frame(state.frame)
            canvas.wait_for_space_press("look again")
        else:
            return state

def set_board(state, target_board, title):
    canvas.set_text(title, "Planning motion...")
    canvas.set_board(target_board)
    canvas.set_detected_board(state.board)
    canvas.set_frame(state.frame)
    canvas.display()

    while True:
        transfer = planner.plan_transfer(state, target_board)
        if transfer is None:
            return state._replace(board=target_board)

        configurations = planner.make_transfer(state, transfer)
        try:
            canvas.set_text(title, "Moving piece...")
            for configuration in configurations:
                canvas.set_robot_configuration(configuration, planner.camera_to_base)
                canvas.display()
        except (Quit, Reset):
            for configuration in configurations:
                pass
            raise

        state = get_state(state.board, is_setup=True)
        canvas.set_text(title, "Planning motion...")
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.display()

def get_outcome_message(board: chess.Board):
    outcome = board.outcome(claim_draw=True)

    if outcome.winner == chess.WHITE:
        return "The Bishop won"
    if outcome.winner == chess.BLACK:
        return "You won"

    return "It's a draw"


def take_turns(state):
    while not is_game_over(state.board):
        state.board.turn = chess.BLACK
        canvas.set_text("Your turn", "Make your move on the board.")
        canvas.set_board(state.board)
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.wait_for_space_press("end turn")
        state = get_state(state.board)

        state.board.turn = chess.WHITE
        canvas.set_text("The Bishop's turn", "Thinking...")
        canvas.set_board(state.board)
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.display()
        state = set_board(state, chess_bot.play(state.board), "The Bishop's turn")

    canvas.set_text(get_outcome_message(state.board))
    canvas.set_board(state.board)
    canvas.set_frame(state.frame)
    canvas.display()
    canvas.wait()


if __name__ == "__main__":
    try:
        state = set_viewing_pose()
        reset_requested = False

        while True:
            try:
                if reset_requested:
                    reset_requested = False
                    state = get_state(state.board, is_setup=True)
                    state = set_board(state, chess.Board(), "Resetting the board")

                take_turns(state)
            except Reset:
                reset_requested = True

    except Quit:
        pass

    finally:
        canvas.quit()
        vision.quit()
        chess_bot.quit()
        robot.quit()
