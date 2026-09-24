import chess.engine

class ChessBot:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("instinct-chess-bot")

    def play(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=1))
        board = board.copy()
        board.push(result.move)
        return board

    def quit(self):
        self.engine.quit()