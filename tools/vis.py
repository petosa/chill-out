# Add cwd to script path
import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import cv2
import util

from skimage import io


# Get policy
session = sys.argv[1]
log = session + "/log.txt"
print(log)
policy = None
for l in open(log, "r"):
    l = l.replace("\n","")
    if l.startswith("("):
        tuples = l.split("->")
        policy = [eval(t)[0] for t in tuples]
        
if policy is None:
    policy = [[True]*8]
for p in policy: print(p)

# Draw policy
lwidth, lheight = 700, 200
layers = len(policy[0])
canvas = np.zeros((layers*lheight, lwidth, 3))
url = "https://cdn3.iconfinder.com/data/icons/wpzoom-developer-icon-set/500/102-128.png"
lock = io.imread(url)[:,:,:-1]

def draw(i):
    canvas = np.zeros((layers*lheight, lwidth, 3))
    p = policy[i]
    for j in range(len(p)):
        layer = np.ones((lheight-10, lwidth-2, 3)) * (1 if p[j] else .3)
        if p[j]: layer[:,:,:-1] *= .3
        canvas[j*lheight-1:j*lheight+5,1:lwidth-1,:] = 1
        canvas[5+j*lheight:(j+1)*lheight-5,1:lwidth-1,:] = layer
        if not p[j]: canvas[35+j*lheight:35+j*lheight+128, 20:148] += lock
        
    canvas = canvas/canvas.max()
    canvas =cv2.putText(np.copy(canvas), text=str(i+1), org=(550,1580),fontFace=2, fontScale=4, color=(1,1,1), thickness=6)
    return canvas

for i in range(len(policy)):
    img = draw(i)
    #plt.imsave(img)
    # plt.show()

    plt.imsave(str(i), img)
