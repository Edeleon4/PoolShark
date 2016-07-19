# PoolShark

Data types:
  Loc:      A tuple (x, y).
            The center of the table is (0,0).
            Up and to the right is positive.
            We will measure in millimeters (to avoid underflow errors).
            Pool tables apparently come in different sizes, so someone needs to look up how large ours is.
            Pocket sizes we also need to measure
            Pool balls are 61.5 mm diameter.
            
  Ball loc: The center of the ball.
  Pocket loc: the center of the pocket.
  Bucketing: A dict of loc->bool, using the coordinates of the bottom-left corner of the bucket loc.
             So for example:  BUCKETING[(150, -150)] = 1

Input                         Output                        Person
--------------------------------------------------------------------------------------
Picture of pool table         Non-skewed picture            Eddy's magic phone thing

Non-skewed picture            Dict of ball locs             Hunter/Ilya

Dict of ball coordinates      Potentially-colliding         Ratan
                                ball locs           

Dict of ball coordinates,     Target ball loc list          Ratan
  target ball ID       

Dict of ball coordinates      Binary ball bucketing         Ratan

None                          Expanded pocket locs          Ratan

Pocket locs, cue loc,         Potentially-hittable          Eddy
  and target ball loc           pocket locs

Origin loc, end loc           Angle necessary               Eddy

Pocket loc, cue loc,          Cue ball loc during           Eddy
  target ball loc               collision
  
Origin loc, end loc,          Binary path-has-collision     ????
  ball bucketing
  
Origin loc, end loc           Path score                    ????

Put it all together           Angle of shot                 ????

Angle of shot                 Picture with shot drawn in    ????
