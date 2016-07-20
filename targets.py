def normFunc(items):
		item_sum = float(sum(map(abs,items)))
		return lambda x : float(x)/item_sum

def cueball_collision_loc(cueball,pocket, target):
		rad = cueball[2]
		diam = 2*rad

		# distance between the balls scaling delta to oen ball 			  
		# hypotenuse along the impact path
		delta_x = pocket[0] - target[0]
		delta_y = pocket[1] - target[1]

		norm = normFunc([delta_x,delta_y])
		x_shift = norm(delta_x)*diam
		y_shift =  norm(delta_y)*diam
	
		hit_pos_x = target[0] + x_shift
		hit_pos_y = target[1] + y_shift
		# Hacky way of dealing with relative angles
		# gaurantees the resulting ball is further from the pocket

		if abs(hit_pos_x - pocket[0]) < abs(target[0] - pocket[0]):
				hit_pos_x = target[0] - x_shift
						
		if abs(hit_pos_y - pocket[1]) < abs(target[1] - pocket[1]):
				hit_pos_y = target[1] - y_shift

		ans = [hit_pos_x, hit_pos_y]
		print ans	
		return (hit_pos_x,hit_pos_y)

class Ball:
		def __init__(self,x,y):
				self.x = float(x)
				self.y = float(y)
				self.rad = 1.0
		
import matplotlib.pyplot as plt
def debug():
		pocket = (2,4,1) 
		target = (-3,-10,1) 
		locs = cueball_collision_loc((1,1,1),pocket,target)

		coord = [pocket,target,locs]



		print "pocket ro, target blue, loc green"
		plt.plot([coord[0][0]],[coord[0][1]], 'ro')
		plt.plot([coord[1][0]],[coord[1][1]], 'bs')
		plt.plot([coord[2][0]],[coord[2][1]], 'g^')
		plt.axis([-30, 30, -30, 30])
		plt.show()

