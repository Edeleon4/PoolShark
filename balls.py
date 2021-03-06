class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Ball:
    def __init__(self, x, y):
        self.diameter = 0.1
        self.x = x
        self.y = y
        self.coord = Coord(x,y)

class ObjectBall(Ball):
    pass

class One(ObjectBall):
    pass

class Two(ObjectBall):
    pass

class Three(ObjectBall):
    pass

class Four(ObjectBall):
    pass

class Five(ObjectBall):
    pass

class Six(ObjectBall):
    pass

class Seven(ObjectBall):
    pass

class Eight(ObjectBall):
    pass

class Nine(ObjectBall):
    pass

class CueBall(Ball):
    pass
