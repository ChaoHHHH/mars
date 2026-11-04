EPS0  = 8.8541878188E-12 # F/m

class InsulatingLayer:
    def __init__(self, eps_r, tox) -> None:
        self.eps_r = eps_r
        self.eps = eps_r * EPS0
        self.tox = tox