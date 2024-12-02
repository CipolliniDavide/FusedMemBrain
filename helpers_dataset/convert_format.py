from .load_dataset import data_loader
from Class_SciPySparseV2.utils import utils

import numpy as np


def convert_format_of_dataset(dataset_lst: list,
                              net_param: dict,
                              n_input: int = 20,
                              linear_fraction_of_inner_square=7,
                              show_info: bool = False,
                              ):
    """
    Converts the superpixel dataset into location of the n_input input electrodes.
    The inner square is centered in the square network and is of

    :param dataset_name: name of the dataset to be loaded.
    :param n_input: number of electrodes acting as input electrodes.
    :param linear_fraction_of_inner_square: controls the size of the inner square where electrodes will be placed according to
        inner-square-linear-size=(network-linear-size)/linear_fraction_of_inner_square.
    :return:
        X: features of each node that are then used as applied voltages
        Y: class of each sample.
        coord_electrodes: the location of the electrodes.
    """

    # frac_data = 1
    #
    # ind_start_tr = 0
    # ind_end_tr = int(len(dataset_lst[0]) * frac_data)

    # ind_start_test = 0
    # ind_end_test = int(len(test_set) * frac_data)

    X = []
    Y = []
    coord_electrodes = []
    for set in dataset_lst:
        # X.append(np.array([data.x.numpy().reshape(-1) for data in set[ind_start_tr:ind_end_tr]]))
        Y.append(np.array([data.y.numpy() for data in set]).reshape(-1))
        for data in set:
            pos = data.pos.numpy()
            pos[:, [1, 0]] = pos[:, [0, 1]]
            # Instantiate the input nodes
            pos[:, 0] = utils.scale(pos[:, 0],
                                    out_range=(net_param.rows / linear_fraction_of_inner_square,
                                               (linear_fraction_of_inner_square-1) * net_param.rows / linear_fraction_of_inner_square - 1))
            pos[:, 1] = utils.scale(pos[:, 1],
                                    out_range=(net_param.rows / linear_fraction_of_inner_square,
                                              (linear_fraction_of_inner_square-1) * net_param.rows / linear_fraction_of_inner_square - 1))
            # pos[:, 0] = utils.scale(pos[:, 0], out_range=(net_param.rows/4, 3*net_param.rows/4 - 1))
            # pos[:, 1] = utils.scale(pos[:, 1], out_range=(net_param.rows/4, 3*net_param.rows/4 - 1))
            # pos[:, 0] = utils.scale(pos[:, 0], out_range=(0, net_param.rows-1))
            # pos[:, 1] = utils.scale(pos[:, 1], out_range=(0, net_param.rows-1))

            pos_electrodes = np.round(pos, decimals=0)
            # indices = (data.x.numpy() > 0).reshape(-1)
             # Get the indices of the 24 highest values
            indices = np.argpartition(data.x.numpy().reshape(-1), -n_input)[-n_input:]
            indices = indices[np.argsort(data.x.numpy().reshape(-1)[indices])][::-1]
            pos_electrodes = pos_electrodes[indices]
            coord_electrodes.append(pos_electrodes)
            X.append(data.x.numpy().reshape(-1)[indices])
    X = np.reshape(X, (len(X), X[0].shape[0]))
    # X = np.row_stack((X[0], X[1]))
    if len(Y) > 1:
        Y = np.concatenate((Y[0], Y[1]))
    else:
        Y = Y[0]
    # num = 000
    # plt.scatter(coord_electrodes[num][:, 0], coord_electrodes[num][:, 1], c=X[num])
    # plt.title(Y[num])
    # plt.show()

    return X, Y, coord_electrodes
