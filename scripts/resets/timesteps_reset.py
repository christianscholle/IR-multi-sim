from scripts.resets.reset import Reset

class TimestepsReset(Reset):
    def __init__(self, max: int, min: int=None, reward: float=0) -> None:
        super().__init__(reward)
        self.max = max
        self.min = min