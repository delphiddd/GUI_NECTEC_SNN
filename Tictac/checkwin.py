# checkwin.py
# Logic ตรวจผู้ชนะ + บังคับสลับตาผู้เล่น (X / O) ทีละช่อง

# ตำแหน่งที่ชนะทั้งหมด (index 0-8 บนกระดาน 3x3)
WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),   # แถว
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # คอลัมน์
        (0, 4, 8), (2, 4, 6)]              # ทแยง

EMPTY = " "


def check_win(board, player):
    """ชนะก็ต่อเมื่อ player (X หรือ O) เรียงครบ 3 ช่องในแนวใดแนวหนึ่ง"""
    return any(all(board[i] == player for i in line) for line in WINS)


def is_full(board):
    """กระดานเต็มหมดแล้ว (ใช้เช็คเสมอ)"""
    return all(cell != EMPTY for cell in board)


def is_valid_move(board, idx):
    """ลงได้เฉพาะช่อง 0-8 ที่ยังว่าง"""
    return 0 <= idx < 9 and board[idx] == EMPTY


def make_move(board, idx, player):
    """
    ลงหมากให้ player ที่ช่อง idx
    return True ถ้าลงสำเร็จ, False ถ้าช่องไม่ว่าง/นอกกระดาน
    """
    if not is_valid_move(board, idx):
        return False
    board[idx] = player
    return True


class TicTacToe:
    """
    State ของเกมสำหรับให้ GUI เรียกใช้ (ไม่มี loop / input — GUI คุม event เอง)

    วิธีใช้ใน GUI:
        game = TicTacToe()
        result = game.play(idx)          # ตอนผู้เล่นคลิกช่อง idx (0-8)
        if not result["ok"]:             # ช่องไม่ว่าง/จบเกมแล้ว → ไม่ทำอะไร
            return
        # อัปเดตปุ่มช่อง idx เป็น result["player"]
        if result["winner"]:             # มีผู้ชนะ → โชว์ข้อความ + ล็อกกระดาน
            ...
        elif result["draw"]:             # เสมอ
            ...
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """เริ่มเกมใหม่ — เรียกตอนกดปุ่ม Restart"""
        self.board = [EMPTY] * 9
        self.current = "X"        # X เริ่มก่อน
        self.winner = None        # "X" / "O" / None
        self.over = False         # จบเกมหรือยัง (ชนะหรือเสมอ)

    def play(self, idx):
        """
        ลงหมากของผู้เล่นปัจจุบันที่ช่อง idx แล้วสลับตาให้อัตโนมัติ
        return dict สรุปผลให้ GUI เอาไปอัปเดตหน้าจอ
        """
        # จบเกมแล้ว หรือ ช่องไม่ว่าง → ไม่นับ (ตาเดิม ไม่สลับ)
        if self.over or not make_move(self.board, idx, self.current):
            return {"ok": False, "player": None, "winner": None, "draw": False}

        played = self.current     # คนที่เพิ่งลง

        # เช็คผู้ชนะหลังลงสำเร็จ
        if check_win(self.board, played):
            self.winner = played
            self.over = True
            return {"ok": True, "player": played, "winner": played, "draw": False}

        # เช็คเสมอ (กระดานเต็มแต่ไม่มีใครชนะ)
        if is_full(self.board):
            self.over = True
            return {"ok": True, "player": played, "winner": None, "draw": True}

        # ยังไม่จบ → สลับตา
        self.current = "O" if played == "X" else "X"
        return {"ok": True, "player": played, "winner": None, "draw": False}
