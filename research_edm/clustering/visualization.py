import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# mpl.use('TkAgg')  # for plotting in separate window


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_clustering_points(tx, ty, tz, labels, title, colors_per_class, three_d=False, savefig=False, x_label="", y_label="", z_label=""):
    # initialize matplotlib plot

    fig = plt.figure()
    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel(z_label)
    else:
        ax = fig.add_subplot(111)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.title.set_text(title)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        if three_d:
            current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255  # np.float is deprecated

        # add a scatter plot with the corresponding color and label
        if three_d:
            ax.scatter(current_tx, current_ty, current_tz, c=color, label=label)
        else:
            ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    if not savefig:
        plt.show()

    return plt


def visualize_3d_clustering(clustering, labels, title, colors_per_class, three_d=False, savefig=False, x_label="", y_label="", z_label=""):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = clustering[:, 0]
    ty = clustering[:, 1]
    tz = None
    if three_d:
        tz = clustering[:, 2]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    if three_d:
        tz = scale_to_01_range(tz)

    # visualize the plot: samples as colored points
    return visualize_clustering_points(tx, ty, tz, labels, title, colors_per_class, three_d, savefig=savefig, x_label=x_label, y_label=y_label, z_label=z_label)
