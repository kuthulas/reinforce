import matplotlib.pyplot as plt
import math, random

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labeltop='off', labelleft='off', labelright='off')
ax.grid()

tag = {"bd": 0, "bu": 1, "fu": 2, "fd": 3}

state = random.randint(0,15)
print state+1

xlook = [[tag["bd"],tag["bd"]],
		 [tag["bd"],tag["bu"]],
		 [tag["bd"],tag["fu"]],
		 [tag["bd"],tag["fd"]],
		 [tag["bu"],tag["bd"]],
		 [tag["bu"],tag["bu"]],
		 [tag["bu"],tag["fu"]],
		 [tag["bu"],tag["fd"]],
		 [tag["fu"],tag["bd"]],
		 [tag["fu"],tag["bu"]],
		 [tag["fu"],tag["fu"]],
		 [tag["fu"],tag["fd"]],
		 [tag["fd"],tag["bd"]],
		 [tag["fd"],tag["bu"]],
		 [tag["fd"],tag["fu"]],
		 [tag["fd"],tag["fd"]]]

def get_vector(angles,lengths):
	x = [0,0,0,0]
	y = [0,0,0,0]
	x[1] = lengths[0]*math.cos(angles[0])
	y[1] = lengths[0]*math.sin(angles[0])
	x[2] = x[1] + lengths[1]*math.cos(angles[0]+angles[1])
	y[2] = y[1] + lengths[1]*math.sin(angles[0]+angles[1])
	x[3] = x[2] + lengths[2]*math.cos(angles[0]+angles[1]+angles[2])
	y[3] = y[2] + lengths[2]*math.sin(angles[0]+angles[1]+angles[2])
	return x, y

line0, = ax.plot([], [], 'o-', lw=20)
line1, = ax.plot([], [], 'o-', lw=10)
line2, = ax.plot([], [], 'o-', lw=10)
line3, = ax.plot([], [], 'o-', lw=10)
line4, = ax.plot([], [], 'o-', lw=10)

factor = math.pi/180

angles = [[factor*-120,factor*-20,factor*90], # back down
		  [factor*-130,factor*-50,factor*90], # back up
		  [factor*-30,factor*-90,factor*90], # front up
		  [factor*-60,factor*-20,factor*90]] # front down

lengths = [0.6,0.4,0.2]
X1, Y1 = get_vector(angles[xlook[state][0]], lengths)
X2, Y2 = get_vector(angles[xlook[state][1]], lengths)

line0.set_data([-0.02,-0.02],[0.,.75])
line1.set_data(X1,Y1)
line2.set_data([x - 0.05 for x in X2],Y2)
plt.show()
