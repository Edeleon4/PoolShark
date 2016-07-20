TABLE_LENGTH = 2600
TABLE_WIDTH = TABLE_LENGTH/2
BALL_DIAMETER = 61.5

POCKET_LOCS = [
    (-TABLE_WIDTH/2, 0),
    ( TABLE_WIDTH/2, 0),
    (-TABLE_WIDTH/2, TABLE_LENGTH/2),
    ( TABLE_WIDTH/2, TABLE_LENGTH/2),
    (-TABLE_WIDTH/2, -TABLE_LENGTH/2),
    ( TABLE_WIDTH/2, -TABLE_LENGTH/2),
    ]

BUCKET_PRECISION = 10

class UnimplementedException(Exception):
    pass
def unimplemented(*args):
    raise UnimplementedException()
unskew_table = unimplemented
get_ball_locs = unimplemented
expand_loc = unimplemented
feasible_shot = unimplemented
cueball_collision_loc = unimplemented
get_angle_from_path = unimplemented
get_buckets = unimplemented
path_has_collision = unimplemented
get_shot_difficulty = unimplemented

##from aarti import unskew_table
##from hunter import get_ball_locs
##from ratan import expand_loc
##from someone import get_buckets, path_has_collision
##from someone import get_shot_difficulty
from eddie import cueball_collision_loc, feasible_shot
from jacob_pool import get_angle_from_path

flatten = lambda l:[item for sublist in l for item in sublist]

def get_next_shot(table_pic):

    try:
        # unskew table picture
        unskewed = unskew_table(table_pic)

        # locate each ball on the table
        ball_locs = get_ball_locs(unskewed)
        
    except UnimplementedException:
        ball_locs = {
            0: (0, 0),
            9: (600, 0)
            }


    try:
        # select a target ball (for nine-ball, this must be the lowest ball on the table)
        target_ball_id = min([key for key in ball_locs if key != 0])

        # Get the location of the cueball, all target balls, and all pockets.
        # This includes virtual pockets and target balls, to account for banking off the bumpers
        cueball_loc = ball_locs[0]
        all_target_locs = expand_loc(ball_locs[target_ball_id])
        all_pocket_locs = flatten([expand_loc(pocket_loc) for pocket_loc in POCKET_LOCS])
        
    except UnimplementedException:
        target_ball_id = 9
        cueball_loc = (0,0)
        all_target_locs = [(600, 600)]
        all_pocket_locs = POCKET_LOCS


    try:
        # For each target-pocket pair, find out if it's possible - if it is, figure out where
        #   we need to hit the cueball to make it so
        potential_shots = []
        for target_loc in all_target_locs:
            for pocket_loc in all_pocket_locs:
                if feasible_shot(cueball_loc, target_loc, pocket_loc):
                    cueball_dest = cueball_collision_loc(cueball_loc, target_loc, pocket_loc)
                    potential_shots.append(((cueball_loc, cueball_dest),
                                             (target_loc,  target_dest)))
    except UnimplementedException:
        potential_shots = [(((  0, 0), (538.5, 0)),
                            ((600, 0), ( 1300, 0)))]


    try:
        # Look at all the other balls on the table that might fuck up our shot,
        #   and calculate a coarse set of location buckets for collision detection
        potential_collision_locs = flatten([expand_loc(ball_locs[ball_id]) for ball_id in ball_locs
                                            if ball_id not in [0, target_ball_id]])
        collision_buckets = get_buckets(potential_collision_locs, BUCKET_PRECISION)

    except UnimplementedException:
        potential_collision_locs = []
        collision_buckets = {(x, y):0 for x in range(0, TABLE_WIDTH, BUCKET_PRECISION)
                                      for y in range(0, TABLE_LENGTH, BUCKET_PRECISION)}


    try:
        # Iterate through our potential shots, keeping only the ones that we can make without
        #   hitting another ball en route
        valid_shots = []
        for potential_shot in potential_shots:
            first_leg, second_leg = potential_shot
            if not path_has_collision(first_leg, collision_buckets) and \
               not path_has_collision(second_leg, collision_buckets):
                valid_shots.append((first_leg, second_leg))

    except UnimplementedException:
        valid_shots = potential_shots


    try:
        # For each shot that is makeable, assign an "difficulty" score. Sort by that to select the
        #    easiest shot
        scored_shots = [(shot, get_shot_difficulty(shot)) for shot in valid_shots]
        scored_shots.sort(key=lambda x:x[1])
        best_shot = scored_shots[0]
        
    except UnimplementedException:
        scored_shots = valid_shots
        best_shot = scored_shots[0]


    try:
        # Calculate the angle we need to hit the cue ball at to make our shot happen
        cueball_start_loc, cueball_collision_loc = best_shot[0]
        angle = get_angle_from_path(cueball_start_loc, cueball_collision_loc)
        
    except UnimplementedException:
        cueball_start_loc, cueball_collision_loc = best_shot[0]
        angle = 0.0     

    # Return that shit
    return (cueball_start_loc, angle)

get_next_shot(None)
