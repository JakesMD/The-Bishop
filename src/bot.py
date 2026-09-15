import chess.engine

class Bot:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("instinct-chess-bot")

    def get_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=1))
        return result.move

    def quit(self):
        self.engine.quit()