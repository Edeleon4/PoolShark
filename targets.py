def normFunc(items):
		item_sum = float(sum(items))
		return lambda x : float(x)/item_sum

def cueball_collision_loc(pocket, target):
		rad = target.rad
		diam = 2*rad

		# distance between the balls scaling delta to oen ball 			  
		# hypotenuse along the impact path
		delta_x = pocket.x - target.x
		delta_y = pocket.y - target.y
		norm = normFunc([delta_x,delta_y])
		x_shift = norm(delta_x)*diam
		y_shift =  norm(delta_y)*diam
	
		hit_pos_x = target.x + x_shift
		hit_pos_y = target.y + y_shift
		# Hacky way of dealing with relative angles
		# gaurantees the resulting ball is further from the pocket

		if abs(hit_pos_x - pocket.x) < abs(target.x - pocket.x):
				hit_pos_x = target.x - x_shift
						
		if abs(hit_pos_y - pocket.y) < abs(target.y - pocket.y):
				hit_pos_y = target.y - y_shift

		ans = [hit_pos_x, hit_pos_y]
		print ans	
		return Ball(hit_pos_x,hit_pos_y)

class Ball:
		def __init__(self,x,y):
				self.x = float(x)
				self.y = float(y)
				self.rad = 1.0
		
import matplotlib.pyplot as plt
def debug():
		pocket = Ball(2,4) 
		target = Ball(-3,8) 
		locs = cueball_collision_loc(pocket,target)

		coord = [pocket,target,locs]
		print map(lambda x:x.x,coord)
		print map(lambda x:x.y,coord)



		print "pocket ro, target blue, loc green"
		plt.plot([coord[0].x],[coord[0].y], 'ro')
		plt.plot([coord[1].x],[coord[1].y], 'bs')
		plt.plot([coord[2].x],[coord[2].y], 'g^')
		plt.axis([-30, 30, -30, 30])
		plt.show()

