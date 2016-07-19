def normFunc(items):
		item_sum = sum(items)
		return lambda x : float(x)/item_sum

def cueball_collision_loc(pocket, target):
		rad = target.rad
		diam = 2*rad
		# distance between the balls scaling delta to oen ball 			  # hypotenuse along the impact path
		delta_x = pocket.x - target.x
		delta_y = pocket.y - target.y
		norm = normFunc([delta_x,delta_y])
		x_shift = norm(delta_x)*diam
		y_shift =  norm(delta_y)*diam
	
		hit_pos_x = target.x + x_shift
		hit_pos_y = target.y + y_shift
		# Hacky way of dealing with relative angles
		# gaurantees the resulting ball is further from the pocket

		if (hit_pos_x - pocket.x) < (target.x - pocket.x):
				hit_pos_x = target.x - x_shift
						
		if (hit_pos_y - pocket.y) < (target.y - pocket.y):
				hit_pos_y = target.y - y_shift

		ans = [hit_pos_x, hit_pos_y]
		print ans	
		return ans

class Ball:
		def __init__(self,x,y):
				self.x = x
				self.y = y
				self.rad = 1
		
pocket = Ball(2,4) 
target = Ball(-3,8) 
cueball_collision_loc(pocket,target)
