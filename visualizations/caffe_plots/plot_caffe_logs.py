'''
Produces plots from a pair of caffe.INFO.train and caffe.INFO.test log files at the given path.

Usage:
    caffe_plots.py <path_to_logs>
'''

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt


def plot_from_logs(train_file, valid_file):
    '''
    Training Log Header:
    #Iters Seconds TrainingLoss LearningRate

    Valid Log Header:
    #Iters Seconds TestAccuracy TestLoss

    :param train_file:
    :param valid_file:
    :return:
    '''
    tX = np.loadtxt(train_file, skiprows=1, delimiter=',')
    try:
        t_iters = tX[:, 0]
    except IndexError:
        print "Not enough iterations to plot."
        exit(1)
    seconds = tX[:, 1]
    plt.figure(1)
    plt.subplot(211)
    p1, = plt.plot(t_iters, tX[:, 2], 'b', label="Training Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss/Accuracy')
    plt.legend(bbox_to_anchor=(0.,1.02, 1., 0.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.grid()

    # Learning rate:
    plt.subplot(212)
    p3, = plt.plot(t_iters, tX[:, 3], label="Learning Rate")
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.legend(loc=3)
    plt.grid()
    plt.title('Caffe Model')
    # try Validation plots:
    try:
        vX = np.loadtxt(valid_file, skiprows=1, delimiter=',')
        v_iters = vX[:, 0]
        p2, = plt.plot(v_iters, vX[:, 2], 'g', label="Validation Accuracy")
        p3, = plt.plot(v_iters, vX[:, 3], 'r', label="Validation Loss")
    except IndexError:
        print "Not enough iterations to plot validation."
    plt.show()


if __name__ == "__main__":
    arguments = docopt(__doc__)
    logs_path = arguments['<path_to_logs>']
    train_file = logs_path + 'caffe.INFO.train'
    valid_file = logs_path + 'caffe.INFO.test'
    #plt.style.use('ggplot')
    plot_from_logs(train_file, valid_file)













