from dataclasses import dataclass

from neural_network import neural_network
import numpy as np
from tkinter import *


@dataclass
class Point(object):
    color: int
    x: int = 0
    y: int = 0


class ImageGenerator(object):
    def __init__(self, width, height):
        self.network = neural_network(2, 5, 2, alpha=0.001)
        self.width = width
        self.height = height
        self.image = PhotoImage(width=self.width, height=self.height)

    def _get_image(self) -> PhotoImage:
        pixels = ' '.join('{' + ' '.join(
            App.get_color(0, 0, 255) if self.network.get_ans(np.array([[j, i]]))[0][0] > 0.5
            else App.get_color(0, 255, 0)
            for j in range(self.width)) + '}' for i in range(self.height))
        self.image.put(pixels, (0, 0))
        return self.image

    def update_image(self, points: list) -> PhotoImage:
        x = [np.array([[point.x, point.y]]) for point in points]
        y = [np.array([[0, 1]]) if point.color == 0 else np.array([[1, 0]]) for point in points]
        self.network.study(x, y)
        return self._get_image()


class App(Frame):
    def __init__(self, parent: Tk):
        Frame.__init__(self, parent)
        self.image_generator = ImageGenerator(300, 300)
        self.canvas = Canvas(root, width=300, height=300)
        self.points = []
        self.image_id = self.canvas.create_image(0, 0, image=self.image_generator.update_image(self.points), anchor=NW)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.set_blue_point)
        self.canvas.bind('<Button-3>', self.set_green_point)
        # self.canvas.after(1000, self.show_image)

    def show_image(self) -> None:
        self.image_generator.update_image(self.points)
        # self.canvas.after(1000, self.show_image)

    def set_blue_point(self, event):
        x = event.x
        y = event.y
        self.points.append(Point(0, x, y))
        self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                     fill=App.get_color(0, 255, 0),
                                     outline=App.get_color(255, 255, 255))
        self.show_image()

    def set_green_point(self, event):
        x = event.x
        y = event.y
        self.points.append(Point(1, x, y))
        self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                     fill=App.get_color(0, 0, 255),
                                     outline=App.get_color(255, 255, 255))
        self.show_image()

    @staticmethod
    def get_color(a, b, c):
        return '#%02x%02x%02x' % (a, b, c)


if __name__ == '__main__':
    root = Tk()
    root.title('AI')
    root.geometry('300x300')
    app = App(root)
    root.mainloop()
