import cv2
import numpy as np
import chess.svg
import cairosvg

class Canvas:
    def __init__(self):
        self.window_name = "The Bishop"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _wait_for_key_press(self, character):
        key = None
        while key != ord(character):
            key = cv2.waitKey(1) & 0xFF

    def wait_for_space_press(self):
        self._wait_for_key_press(" ")

    def wait_for_q_press(self):
        self._wait_for_key_press("q")

    def display_board(self, board: chess.Board, frame: np.ndarray, illegal=False):
        board_svg = chess.svg.board(board=board, size=800, flipped=True)
        png_bytes = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))
        board_image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        board_h = board_image.shape[0]
        frame_h, frame_w, _ = frame.shape
        new_frame_w = int(frame_w * (board_h / frame_h))
        resized_frame = cv2.resize(frame, (new_frame_w, board_h))

        combined_body = np.hstack((board_image, resized_frame))

        outcome = board.outcome(claim_draw=True)
        if illegal:
            status_text = "Illegal move! Try again. Press SPACE once done."
        elif outcome is not None:
            if outcome.winner == chess.WHITE:
                status_text = "The Bishop won. Press Q to quit."
            elif outcome.winner == chess.BLACK:
                status_text = "You won. Press Q to quit."
            else:
                status_text = "It's a draw. Press Q to quit."
        elif board.turn == chess.WHITE:
            status_text = "The Bishop's turn."
        else:
            status_text = "Your turn. Press SPACE once done."

        total_width = combined_body.shape[1]
        header = np.zeros((80, total_width, 3), dtype=np.uint8)
        cv2.putText(header, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        final_view = np.vstack((header, combined_body))
        cv2.imshow(self.window_name, final_view)
        cv2.waitKey(1)

    def quit(self):
        cv2.destroyAllWindows()