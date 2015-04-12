import random
import numpy as np

states = np.array([[3,3,0],[3,0,3],[0,3,3]])

for i in range(10):
	state = states[random.randint(0,2)]
	print state
	state[0] = 6