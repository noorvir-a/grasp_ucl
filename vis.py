from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

import numpy as np
import math as m
import csv



class VisKit():

    @staticmethod
    def load_event_file(event_filepath):
        ea = EventAccumulator(event_filepath)
        ea.Reload()

        return ea

    @staticmethod
    def get_lc(ea, scalar_strs):
        """ Takes event accumulator and scalar names as input and return stacked array"""

        train_steps = np.squeeze([np.asarray([scalar.step]) for scalar in ea.Scalars(scalar_strs[0])])
        # time_steps = np.squeeze([np.asarray([scalar.wall_time]) for scalar in ea.Scalars(scalar_strs[0])])
        train_acc = np.squeeze([np.asarray([scalar.value]) for scalar in ea.Scalars(scalar_strs[0])])
        loss = np.squeeze([np.asarray([scalar.value]) for scalar in ea.Scalars(scalar_strs[2])])

        val_steps = np.squeeze([np.asarray([scalar.step]) for scalar in ea.Scalars(scalar_strs[1])])
        val_acc = np.squeeze([np.asarray([scalar.value]) for scalar in ea.Scalars(scalar_strs[1])])

        train_data = np.transpose(np.stack([train_steps, train_acc, loss]))
        val_data = np.transpose(np.stack([val_steps, val_acc]))

        return train_data, val_data

    @staticmethod
    def save_to_csv(data, filepath, header):


        np.savetxt(filepath, data, fmt='%.5f', delimiter=",", header=header)

        # with open(csv_loss_filename, 'w') as csv_loss_file, \
        #         open(csv_eval_filename, 'w') as csv_eval_file:
        #     # Write meta-data and headers to CSV file
        #     csv_writer = csv.writer(csv_loss_file)
        #     csv_writer.writerow(csv_loss_header)
        #
        #     csv_writer = csv.writer(csv_eval_file)
        #     csv_writer.writerow(csv_eval_header)
        pass

    @staticmethod
    def plot():

        pass

# fig, ax = plt.subplots()
# x_axis = range(num_data_points)
# ax.bar(x_axis, histogram_data, width)
#
# #ticks
# if num_data_points <= 10:
#     x_ticks = range(num_data_points)
#     ax.set_xticks([tick + width / 2.0 for tick in x_ticks])
#     ax.set_xticklabels(bins)
# else:
#     x_ticks = np.linspace(0, len(histogram_data)-1, num_ticks)
#     ax.set_xticks([tick for tick in x_ticks])
#     # x_tick_labels = [ if (bin_num + 1) % int() for bin_num, bin in enumerate(bins) ]
#     x_tick_labels = [bins[int(tick)] for tick in x_ticks]
#     ax.set_xticklabels(['{:.1e}'.format(label) for label in x_tick_labels])
#
# ax.set_xlabel('Robust Ferrau Canny Metric')
# ax.set_ylabel('Number of samples')
# plt.show()




if __name__ == '__main__':


    event_filepath = '/home/noorvir/tf_models/GUAN-t/summaries/17-08-08-22:29:07/'
    visk = VisKit()

    event_accumulator = visk.load_event_file(event_filepath)

    train_data, val_data = visk.get_lc(event_accumulator, ['train_accuracy', 'val_accuracy', 'xentropy_loss'])

    t_out_filepath = '/home/noorvir/tf_models/GUAN-t/summaries/17-08-08-22:29:07/train_data.csv'
    v_out_filepath = '/home/noorvir/tf_models/GUAN-t/summaries/17-08-08-22:29:07/val_data.csv'
    visk.save_to_csv(train_data, t_out_filepath, header='step, accuracy , loss')
    visk.save_to_csv(val_data, v_out_filepath, header='step, accuracy')






# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# g1 = np.random.multivariate_normal(np.random.rand(2), np.diag(np.random.rand(2)), 10000)
#
# ax.plot(list(np.random.rand(10000, 2)),g1)
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y = np.mgrid[-1.0:1.0:300j, -1.0:1.0:300j]
# # Need an (N, 2) array of (x, y) pairs.
# xy = np.column_stack([x.flat, y.flat])
# mu = np.array([0.0, 0.0])
# sigma = np.array([.5, .3])
# covariance = np.diag(sigma**2)
# z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# x += 1.2
# y += 1.33
# xy = np.column_stack([x.flat, y.flat])
# z += multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# # Reshape back to a (30, 30) grid.
# z = z.reshape(x.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x,y,z)
# #ax.plot_wireframe(x,y,z)
# plt.show()

