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
  
  Bucketing: A dict of loc->bool, using the coordinates of the bottom-left corner of the bucket loc. So for example:  BUCKETING[(150, -150)] = 1


Input                         |Output                        |Person
------------------------------|------------------------------|---------------
Picture of pool table         |Non-skewed picture            |Eddy's magic phone thing
Non-skewed picture            |Dict of ball locs             |Hunter/Ilya
Dict of ball locs, target ball ID        |Expanded target ball locs          |Ratan
Dict of ball locs, target ball ID        |Expanded ball locs (non-cue, non-target)        |Ratan
Dict of ball locs, target ball ID      |Binary ball bucketing (non-cue, non-target)        |Ratan
None                          |Expanded pocket locs          |Ratan
Pocket locs, cue loc, and target ball loc          |Potentially-hittable   pocket locs        |Eddy
Origin loc, end loc           |Angle necessary               |Eddy
Pocket loc, cue loc, target ball loc      |Cue ball loc during  collision          |Eddy
Origin loc, end loc, ball bucketing          |Binary path-has-collision     |????
Origin loc, end loc           |Path score                    |????
Put it all together           |Angle of shot                 |Jacob
Angle of shot                 |Picture with shot drawn in    |????
