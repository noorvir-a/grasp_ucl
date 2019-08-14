
from perception import BinaryImage, ColorImage, DepthImage, GdImage, GrayscaleImage, RgbdImage, RenderMode
from gqcnn import Visualizer as vis2d
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

import operator

import numpy as np


class UCLVisualiser(object):
    """ Class for useful visualisations """

    # set prettier style
    plt.style.use('ggplot')

    @staticmethod
    def histogram(histogram_data, bins, width=1.0, num_ticks=10):
        """
        Plot histogram from data

        (Inspired by: https://stackoverflow.com/questions/29508208/best-way-to-plot-categorical-data)
        """

        num_data_points = len(histogram_data)

        fig, ax = plt.subplots()
        x_axis = range(num_data_points)
        ax.bar(x_axis, histogram_data, width)

        #ticks
        if num_data_points <= 10:
            x_ticks = range(num_data_points)
            ax.set_xticks([tick + width / 2.0 for tick in x_ticks])
            ax.set_xticklabels(bins)
        else:
            x_ticks = np.linspace(0, len(histogram_data)-1, num_ticks)
            ax.set_xticks([tick for tick in x_ticks])
            # x_tick_labels = [ if (bin_num + 1) % int() for bin_num, bin in enumerate(bins) ]
            x_tick_labels = [bins[int(tick)] for tick in x_ticks]
            ax.set_xticklabels(['{:.1e}'.format(label) for label in x_tick_labels])

        ax.set_xlabel('Robust Ferrau Canny Metric')
        ax.set_ylabel('Number of samples')
        plt.show()


    @staticmethod
    def visualise_grasp():
        pass


    @staticmethod
    def visualise_grasps():

        # self.visualise([depth_img], proj_grasp_img)

        # proj_grasp_img = Grasp2D(grasp_center,
        #                          P_grasp_camera.angle,
        #                          P_grasp_camera.depth,
        #                          width=self.gripper.max_width,
        #                          camera_intr=scaled_camera_intr)
        pass
