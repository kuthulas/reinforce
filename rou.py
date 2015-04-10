import random

probs = [0.1/0.6, 0.25/0.6, 0.6/0.6, 0.05/0.6]
x = [0,0,0,0]

for i in range(1000):
	r = random.random()
	index = 0
	while(r >= 0 and index < len(probs)):
	  r -= probs[index]
	  index += 1
	x[index-1] += 1

print x