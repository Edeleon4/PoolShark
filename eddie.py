import math
BALL_DIAMETER = 61.5
def normFunc(items):
		item_sum = float(sum(map(abs,items)))
		return lambda x : float(x)/item_sum

def cueball_collision_loc(cueball,pocket, target):

		# distance between the balls scaling delta to oen ball 			  
		# hypotenuse along the impact path
		delta_x = pocket[0] - target[0]
		delta_y = pocket[1] - target[1]

		norm = normFunc([delta_x,delta_y])
		x_shift = norm(delta_x)*BALL_DIAMETER
		y_shift =  norm(delta_y)*BALL_DIAMETER
	
		hit_pos_x = target[0] + x_shift
		hit_pos_y = target[1] + y_shift
		# Hacky way of dealing with relative angles
		# guarantees the resulting ball is further from the pocket

		if abs(hit_pos_x - pocket[0]) < abs(target[0] - pocket[0]):
				hit_pos_x = target[0] - x_shift
						
		if abs(hit_pos_y - pocket[1]) < abs(target[1] - pocket[1]):
				hit_pos_y = target[1] - y_shift

		ans = [hit_pos_x, hit_pos_y]
		print ans	
		return (hit_pos_x,hit_pos_y)

def feasible_shot(cueball, pocket, target):
		print cueball
		print target
		min_delta = min(abs(cueball[0]-target[0]),abs(cueball[1]-target[1]))
		angle = find_angle(cueball,target, pocket)
		print angle
		return min_delta > BALL_DIAMETER and angle < 2*math.pi*3.0/4 and angle > math.pi*2.0/4

def find_angle(p0,p1,p2):
		a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
		b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
		c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
		return math.acos( (a+b-c) / math.sqrt(4*a*b) )

class Ball:
		def __init__(self,x,y):
				self.x = float(x)
				self.y = float(y)
				self.rad =BALL_DIAMETER/2 
		
import matplotlib.pyplot as plt
def debug():
		cueball = (-65,-64)
		pocket = (70,1) 
		target = (10,10) 
		locs = cueball_collision_loc(cueball,pocket,target)

		coord = [cueball,locs,target,pocket]


		print feasible_shot(cueball,pocket,target)

		plt.plot([coord[0][0]],[coord[0][1]], 'ro')
		plt.plot([coord[1][0]],[coord[1][1]], 'bs')
		plt.plot([coord[2][0]],[coord[2][1]], 'g^')
		plt.plot([coord[3][0]],[coord[3][1]], 'rs')
		plt.axis([-200, 200, -200, 200])
		plt.show()
