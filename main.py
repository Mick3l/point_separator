from neural_network import neural_network
import numpy as np
from tkinter import *
import random

if __name__ == '__main__':

    network = neural_network(2, 3, 2, alpha=0.001)
    # study
    study_size = 100000
    for i in range(study_size):
        x = np.random.rand(1, 2)
        if x[0][0] > x[0][1]:
            y = np.array([[1, 0]])
        else:
            y = np.array([[0, 1]])
        network.study(x, y)

    root = Tk()
    root.title('AI')
    root.geometry('500x500')
    canvas = Canvas(root, width=500, height=500)
    i = PhotoImage(width=500, height=500)
    colors = [[255, 0, 0] if  for j in range(500 * 500)]
    row = 0
    col = 0
    for color in colors:
        i.put('#%02x%02x%02x' % tuple(color), (row, col))
        col += 1
        if col == 500:
            row += 1
            col = 0
    canvas.pack()
    canvas.create_image(250, 250, image=i)
    root.mainloop()
