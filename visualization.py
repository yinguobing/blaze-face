"""Visualize the anchor and detection boxes."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Visualizer(object):

    def __init__(self, canvas_size):
        # Create a canvas.
        self.fig, self.ax = plt.subplots(1)

        # Create the background.
        self.background = np.ones(canvas_size+(3,))*256

    def draw_boxes(self, boxes, edgecolor='b'):
        # Draw the boxes.
        for box in boxes:
            w = box[3] - box[2]
            h = box[1] - box[0]
            y = box[0]
            x = box[2]
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                     edgecolor=edgecolor, facecolor='none')
            self.ax.add_patch(rect)

    def show(self):
        plt.imshow(self.background)
        plt.show()
