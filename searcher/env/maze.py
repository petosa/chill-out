# Code adopted from https://github.com/safehammad/python-maze/blob/master/maze.py

import numpy as np
import matplotlib.pyplot as plt
from random import Random
import matplotlib.animation as animation
import sys

from .env import Environment

class Maze(Environment):

    def __init__(self, height=20, width=20, seed=0):
        initial_state = (1,0)
        super().__init__(initial_state)
        self.R = Random(seed)
        np.random.seed(seed)
        self.maze = self.gen_maze(height, width)

    def get_children(self, state):
        actions = []
        r, c = state
        N, S, E, W = (r-1,c), (r+1,c), (r,c+1), (r,c-1)
        if self.maze[N] == 0.: actions.append(N)
        if self.maze[S] == 0.: actions.append(S)
        if c != self.maze.shape[1]-1 and self.maze[E] == 0.: actions.append(E)
        if c != 0 and self.maze[W] == 0.: actions.append(W)
        return actions

    def evaluate(self, state):
        bottom_right = (self.maze.shape[0]-2, self.maze.shape[1]-1)
        return abs(state[0]-bottom_right[0]) + abs(state[1]-bottom_right[1])

    def gen_maze(self, height, width):
        sys.setrecursionlimit(2000)
        def inner(height, width):
            height, width = (height // 2) * 2 + 1, (width // 2) * 2 + 1
            Z = np.ones((height, width))
            def carve(y, x):
                Z[y, x] = 0
                yield Z
                neighbours = [(x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
                self.R.shuffle(neighbours)
                for nx,ny in neighbours:
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if Z[ny, nx] == 1:
                        Z[int((y + ny) / 2),int((x + nx) / 2)] = 0
                        for m in carve(ny, nx):
                            yield m
            x, y = np.random.random_integers(0, width // 2 - 1) * 2 + 1, np.random.random_integers(0, height // 2 - 1) * 2 + 1
            for m in carve(y, x):
                yield m
            Z[1, 0] = Z[-2, -1] = 0
            yield Z
        for m in inner(height, width): pass
        return m

    def visualize(self, explored=[], title="Maze", delay=0):
        cp = np.stack([(-self.maze.copy()+1).tolist()]*3, axis=-1)
        fig, ax = plt.subplots()
        img = ax.imshow(cp, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.title(title)
        self.last_frame = -1
        self.color = None

        def animate(i):
            x,y=explored[i]
            # Only reset color if we are at start state and frame number did not get reset (resizing glitch).
            if x==1 and y==0 and self.last_frame < i: self.color=np.random.random(3)/2+.25
            cp[x,y,:] = self.color
            img.set_data(cp)
            self.last_frame = i
            return img,

        ani = animation.FuncAnimation(fig, animate, interval=delay, blit=True, save_count=0, frames=len(explored), repeat=False)
        plt.show()
