import math

def get_angle_from_path(path):
    start_loc, end_loc = path
    dx = end_loc[0]-start_loc[0]
    dy = end_loc[1]-start_loc[1]
    return math.degrees(math.atan2(dy, dx))
    
assert(get_angle_from_path(((0, 0), (1, 0))) == 0)
assert(get_angle_from_path(((0, 0), (0, 1))) == 90)
assert(get_angle_from_path(((0, 0), (-1, 0))) == 180)
assert(get_angle_from_path(((0, 0), (0, -1))) == -90)
