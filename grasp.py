"""
author: Noorvir Aulakh
date: 17 July 2017

N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
arXiv preprint arXiv:1703.09312 (2017).

"""

from gqcnn import GQCNN, SGDOptimizer
from autolab_core import YamlConfig
import logging
import time

DIR_PATH = r'~/catkin_ws/src/'


def get_elapsed_time(time_in_seconds):
    """ Helper function to get elapsed time """
    if time_in_seconds < 60:
        return '%.1f seconds' % time_in_seconds
    elif time_in_seconds < 3600:
        return '%.1f minutes' % (time_in_seconds / 60)
    else:
        return '%.1f hours' % (time_in_seconds / 3600)


if __name__ == 'main':

    # initialise logger
    logging.getLogger().setLevel(logging.INFO)

    # load config files
    train_config = YamlConfig(DIR_PATH + 'gqcnn/cfg/tools/training.yaml')
    gqcnn_config = train_config['gqcnn_config']

    start_time = time.time()
    gqcnn = GQCNN(gqcnn_config)
    sgd_optimser = SGDOptimizer(gqcnn, train_config)

    with gqcnn.get_tf_graph().as_default():
        sgd_optimser.optimize()

    logging.info('Training Time: ' + str(get_elapsed_time(time.time() - start_time)))


