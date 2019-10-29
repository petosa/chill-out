import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.animation as animation

session = sys.argv[1]
log = session + "/log.txt"
policy = None
for l in open(log, "r"):
    l = l.replace("\n","")
    if l.startswith("("):
        tuples = l.split("->")
        policy = [eval(t)[0] for t in tuples]
        
for p in policy: print(p)

lwidth, lheight = 40, 10

canvas = np.zeros((len(policy[0])*lheight, lwidth, 3))
fig, ax = plt.subplots()
img = ax.imshow(canvas, interpolation='nearest')
plt.xticks([]), plt.yticks([])
def animate(i):
    canvas = np.zeros((len(policy[0])*lheight, lwidth, 3))
    p = policy[i]
    for i in range(len(p)):
        layer = np.ones((lheight-2, lwidth-2, 3)) * (1 if p[i] else .3)
        canvas[1+i*lheight:(i+1)*lheight-1,1:lwidth-1,:] = layer
    img.set_data(canvas)
    return img,

ani = animation.FuncAnimation(fig, animate, interval=800, blit=True, save_count=0, frames=len(policy), repeat=False)
plt.show()
