import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import os
import random

from Class_SciPySparseV2.MemNet_gndOut import MemNet
from Class_SciPySparseV2.ControlSignal import ControlSignal
from Class_SciPySparseV2.visualize import visualize
from Class_SciPySparseV2.visual_utils import set_ticks_label, set_legend, create_colorbar
from Class_SciPySparseV2.utils import utils

from RunModules.run_oscillatory_machine import run_oscillatory_machine as run_machine

import random
import math

def generate_random_points(N, outer_square_linsize, inner_square_linsize, seed=None):
    """
    Generates a list of N random 2D integer coordinates within a specified region.

    Parameters:
    - N (int): The number of random points to generate.
    - outer_square_linsize (int): The linear size of the outer square.
    - inner_square_linsize (int): The linear size of the inner square.
    - seed (int, optional): The random seed for reproducibility.

    Returns:
    - points (list of tuples): A list of N random 2D integer coordinates, where each coordinate is a tuple (x, y).

    The function generates random integer coordinates (x, y) that are within the outer square defined by
    (-outer_square_linsize/2, -outer_square_linsize/2) as the bottom-left corner and (outer_square_linsize/2, outer_square_linsize/2)
    as the top-right corner. The coordinates are not within the inner square defined by
    (-inner_square_linsize/2, -inner_square_linsize/2) as the bottom-left corner and (inner_square_linsize/2, inner_square_linsize/2)
    as the top-right corner, which is centered within the outer square.
    """
    if inner_square_linsize >= outer_square_linsize:
        raise ValueError("Inner square size must be smaller than outer square size")

    if seed is not None:
        random.seed(seed)

    outer_half_size = outer_square_linsize // 2
    inner_half_size = inner_square_linsize // 2

    points = []

    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    while len(points) < N:
        x = random.randint(-outer_half_size, outer_half_size)
        y = random.randint(-outer_half_size, outer_half_size)

        # Check if the point is inside the inner square
        if inner_half_size >= abs(x) >= -inner_half_size and inner_half_size >= abs(y) >= -inner_half_size:
            continue

        valid_point = True

        # Check if the point is at least 1.5 units away from existing points
        for existing_point in points:
            if distance(existing_point, (x + outer_half_size, y + outer_half_size)) < 1.5:
                valid_point = False
                break

        if valid_point:
            points.append((x + outer_half_size, y + outer_half_size))

    return points





def node_map(x, y, rows, cols):
    '''
    Maps 2D euclidean node coordinates to the node index

    Args:
        x: coordinate
        y: coordinate
        rows: # of rows in the net
        cols: # of rows in the net

    Returns: integer label/index of the node

    '''
    node = rows * (y + 1) - (x + 1)
    return int(node)


def refill_list_with_non_overlapping_elements(hidden_src, input_src, N):
    """
    Makes sure that all elements in the hidden list are not contained in input_src.
    Otherwise adds elements to the hidden_src in the range [0, N-1].
    """
    hidden_src_temp = [i for i in hidden_src if i not in input_src]
    hidden_src = add_unique_random_integers(hidden_src_temp=hidden_src_temp,
                                            input_src=input_src,
                                            n=len(hidden_src) - len(hidden_src_temp),
                                            N=N)
    return hidden_src


def add_unique_random_integers(hidden_src_temp, input_src, n, N):
    """
    Helper function for method refill_list_with_non_overlapping_elements().
    """
    np.random.seed(1)

    while n > 0:
        new_int = random.randint(0, N)
        # Check if the new integer is not in hidden_src_temp and also not in input_src
        if new_int not in hidden_src_temp and new_int not in input_src:
            hidden_src_temp.append(new_int)
            n -= 1

    return hidden_src_temp


class Reservoir(object):

    def __init__(self,
                 sim_param=None,
                 osc_param=None,
                 mem_param=None,
                 net_param=None,
                 save_path=None,
                 index_startSaving_index = 0
                 ):

        if sim_param and osc_param and mem_param and net_param:
            self.sim_param = sim_param
            self.osc_param = osc_param
            self.mem_param = mem_param
            self.net_param = net_param

        self.save_path = save_path
        self.index_startSaving_index = index_startSaving_index

        self.run_machine = run_machine
        self.res_state_names_list = ['inp_curr_rec', 'hid_curr_rec', 'V_hid_rec']

        self.visual = Visual_methods()
        if sim_param:
            self.time = np.arange(0, self.sim_param.T + 1/self.sim_param.sampling_rate,
                                  1/self.sim_param.sampling_rate)
        self.save_data_reservoir_states = save_path

    def save_param(self, save_path):
        for d, name in zip([self.sim_param, self.osc_param, self.mem_param, self.net_param],
                           ['sim_param', 'osc_param', 'mem_param', 'net_param']):
            np.save(file=save_path + f'{name}.npy', arr=d)
            utils.save_dict_to_txt(dict=d, savepath=save_path, name=name)
            # Load
            # read_dictionary = np.load(save_data_sim + 'sim_param.npy', allow_pickle='TRUE').item()

    def load_param_from_folder(self, load_path):
        from easydict import EasyDict as edict
        import ast

        for name in ['sim_param', 'osc_param', 'mem_param', 'net_param']:
            with open(load_path + f'{name}.txt', 'r') as file:
                data_str = file.read()
                data_dict = ast.literal_eval(data_str)
                data_dict = edict(data_dict)

            if name == 'sim_param':
                self.sim_param = data_dict #edict(np.load(file=load_path + f'{name}.npy', allow_pickle=True))
                self.time = np.arange(0, self.sim_param.T + 1 / self.sim_param.sampling_rate,
                                      1 / self.sim_param.sampling_rate)
            elif name == 'osc_param':
                self.osc_param = data_dict #p.load(file=load_path + f'{name}.npy', allow_pickle=True)
            elif name == 'mem_param':
                self.mem_param = data_dict  #np.load(file=load_path + f'{name}.npy', allow_pickle=True)
            elif name == 'net_param':
                self.net_param = data_dict # np.load(file=load_path + f'{name}.npy', allow_pickle=True)
            else:
                print('No params with this name.')

    def save_data(self, save_path, list_of_states_arr, hidden_src, src, Y, index_start=0):

        for arr, arr_name in zip(list_of_states_arr, self.res_state_names_list):
            np.save(arr=arr[:, index_start:], file=save_path + f'{arr_name}.npy')
        # np.save(arr=Y_rec[index_start:], file=save_path + 'Y.npy')
        # np.save(arr=node_voltage_list[index_start:], file=save_path + 'node_voltage_list.npy')
        np.save(arr=hidden_src, file=save_path + 'hidden_src.npy')
        np.save(arr=src, file=save_path + 'src.npy')
        np.save(arr=Y, file=save_path + 'label.npy')


    def compute_reservoir_states(self, dataset, ind_start=0, ind_end=None, save_path=None, save=True,
                                 show_electr_location=False, seed=1):
        """
        :param dataset: dataset.
        :param ind_start: start index of samples in dataset.
        :param ind_end: end index of samples in dataset.
        :param save_path: path where to save the reservoir states.
        :param save: if True save the reservoir states after sample presentation. Else it returns the states.
        """
        np.seed = seed
        self.save_data_reservoir_states = save_path

        if not ind_end:
            ind_end = len(dataset)
        print(dataset.Y[:ind_end])

        # num = 0000
        # plt.scatter(dataset.coord_inp_electrodes[num][:, 0], dataset.coord_inp_electrodes[num][:, 1], c=dataset.X[num])
        # plt.title(dataset.Y[num])
        # plt.show()

        # Create ground node.
        gnd = [self.net_param.rows ** 2]

        # Create hidden nodes (i.e., output electrodes) and displace them in the outer square.
        # Note that this is related to the size of inner square that is used to place input electrodes when
        # the dataset is created.
        random_points = generate_random_points(N=60, outer_square_linsize=self.net_param.rows,
                                               inner_square_linsize=(self.net_param.rows)/2 + 4, #
                                               seed=seed)
        hidden_src = sorted([node_map(x, y, rows=self.net_param.rows, cols=self.net_param.cols) for x, y in random_points])
        # plt.scatter([point[0] for point in random_points], [point[1] for point in random_points])
        # plt.show()

        for idx in tqdm(range(ind_start, ind_end), desc="Creating signals for offline training"):

            pos_electrodes = dataset.coord_inp_electrodes[idx]
            intensity = dataset.X[idx]
            label = dataset.Y[idx]

            # Set input electrodes
            input_src = [node_map(x, y, rows=self.net_param.rows, cols=self.net_param.cols) for x, y in pos_electrodes]

            # Following line is necessary only when the region where input electrodes are placed overlaps with the
            # region where output electrodes (also named hidden are placed).
            # TODO: Doesn't take care of inner- or outer- square when replacing elements into hidden_src list
            hidden_src = refill_list_with_non_overlapping_elements(hidden_src=hidden_src, input_src=input_src, N=self.net_param.rows**2)
            # print(np.intersect1d(hidden_src, input_src))

            # Put together all electrodes.
            src = sorted(input_src + hidden_src)

            # Create node labels
            self.input_source_labels = [(input_src[i], 'Inp{:d}'.format(input_src[i])) for i in range(len(input_src))]
            self.hidden_source_labels = [(hidden_src[i], 'n{:d}'.format(hidden_src[i])) for i in range(len(hidden_src))]
            self.node_labels = [(gnd[0], 'Gnd')] + self.input_source_labels + self.hidden_source_labels

            # Define the input signal.
            # For instance a kick of time length proportional to the intensity of each input node feature.
            inputsignal = ControlSignal(sources=src, sim_param=self.sim_param)
            for i, inp in enumerate(input_src):
                # kick = int(50 * data.x.numpy()[i])
                kick = int(50 * intensity[i])
                # kick = len(inputsignal.V_list[0])
                inputsignal.V_list[src.index(inp), 0:kick] = np.ones(kick) * self.sim_param.Vbias * intensity[i]

            # Instantiate memristor network
            self.net = MemNet(mem_param=self.mem_param, net_param=self.net_param, gnd=gnd, src=src,
                              hidden_src=hidden_src, diag=self.net_param.random_diagonals)
            self.net.coordinates = [(node, (feat['coord'])) for node, feat in self.net.G.nodes(data=True)]

            # To visualize location of electrodes.
            if show_electr_location:
                self.visual.disposition_of_electrodes(self.net, input_src, hidden_src, c=intensity, title=label)
                if save_path:
                    plt.savefig(self.save_path+f'el_disp{label}.png')
                plt.show()
                plt.close('all')

            # Start Simulation
            X_rec, Y_rec, hid_curr_rec, inp_curr_rec, H_list, node_voltage_list = (
                self.run_machine(net=self.net,
                                 inputsignal=inputsignal,
                                 hidden_src=hidden_src,
                                 src=src,
                                 gnd=gnd,
                                 osc_param=self.osc_param,
                                 id_list=None,
                                 ind_time_start=0))

            if save:
                save_temp = '{:s}{:06d}/'.format(self.save_data_reservoir_states, idx)
                utils.ensure_dir(save_temp)
                self.save_data(save_path=save_temp,
                               list_of_states_arr=[inp_curr_rec, hid_curr_rec, X_rec],
                               hidden_src=hidden_src,
                               src=src,
                               Y=label,
                               index_start=self.index_startSaving_index)

        print('Presentation of dataset to reservoir is concluded.')
        if save:
            utils.ensure_dir(self.save_path)
            self.save_param(save_path=self.save_path)
            print(f'Data saved data to:\n\t{self.save_path}.\n')
        else:
            return X_rec, Y_rec, hid_curr_rec, inp_curr_rec, H_list, node_voltage_list

    def load_states_from_folder(self, load_path, res_state_name):

        def find_files_in_folder(folder_path, target_file_name):
            file_paths = []

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file == target_file_name:
                        file_paths.append(os.path.join(root, file))

            return file_paths

        return [np.load(f) for f in find_files_in_folder(folder_path=load_path,
                                                         target_file_name=f'{res_state_name}.npy')]

class Visual_methods(object):
    def __init__(self):
        pass

    def grid_of_nxn_signals(self, array,  y_label='', time=None, n=5):
        # Create a 5x5 grid of subplots with shared x and y axes
        if time is not None:
            pass
        else:
            time = np.arange(0, len(array[0]))
        fig, axs = plt.subplots(n, n, sharex='all', sharey='all', figsize=(10, 10))

        array_temp = array[:n**2]
        # Loop through the subplots and plot the rows of the input array
        c = 0
        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                ax.plot(time, array_temp[c, :])
                ax.set_title(f'{c}')
                ax.grid(True)
                c = c + 1

        # Set common labels for the entire grid
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$\mathbf{Time}$')
        plt.ylabel(y_label)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plots
        # plt.show()

    def disposition_of_electrodes(self, net, input_src, hidden_src, c=None, title=''):
        color_node_vis = np.zeros(net.number_of_nodes)
        if c is None:
            color_node_vis[input_src] = 10
        else:
            color_node_vis[input_src] = utils.scale(c, out_range=(10, 15))

        color_node_vis[hidden_src] = 5

        # visualize.plot_network(G=net.G, numeric_label=True, labels=node_labels, figsize=(8,8), node_size=10)
        visualize.plot_network(G=net.G, numeric_label=False, nodes_cmap=plt.cm.inferno,
                               # labels=node_labels,
                               figsize=(8, 8), node_size=80,
                               node_color=color_node_vis)
        plt.title(title)





# data = dataset[idx]

            #
            # pos = data.pos.numpy()
            # pos[:, [1, 0]] = pos[:, [0, 1]]
            # # Input signal class
            # # Instantiate the input nodes
            # pos[:, 0] = utils.scale(pos[:, 0], out_range=(0, self.net_param.rows - 1))
            # pos[:, 1] = utils.scale(pos[:, 1], out_range=(0, self.net_param.rows - 1))
            # pos_electrodes = np.round(pos, decimals=0)
            # pos_electrodes = pos_electrodes[(data.x.numpy() > .4).reshape(-1)]