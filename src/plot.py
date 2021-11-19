import matplotlib.pyplot as plt


def plot(xy, z, xy_pred, z_pred, n_dim):
    """plot real data and predicted data
    all input data are n*1 shape

    Args:
        xy (np.ndarray): real data x and y coordination
        z (np.ndarray): real data rss
        xy_pred (np.ndarray): prediction x and y coordication
        z_pred (np.ndarray): prediction rss
        n_dim (int): plot the specific dimension.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot3D(x, y, z, 'gray')
    x_axis = xy[:, 0]
    y_axis = xy[:, 1]
    ax.scatter(x_axis, y_axis, z[:, n_dim], c=z[:, n_dim],
               cmap='Greys', label=f"WAP00{n_dim+1}")
    if xy_pred is not None:
        ax.scatter(xy_pred[:, 0], xy_pred[:, 1], z_pred[:, n_dim],
                   c=z_pred[:, n_dim], cmap='hsv', label="pred")
    plt.legend(loc='upper left')
    plt.show()