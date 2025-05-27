'''
This class creates a net of MEMRISTORS & RESISTORS: memristors are modeled according to Miranda model.
Code uses SciPy csr_matrix to take advantage of sparseness and symmetry of the matrix.
Computation is made only on the upper triangular matrix.
Diagonals are supposed to be zero (no self link).

These script are made to have the ground node outside the network. This means that each electrode has one terminal
linked to the network and one terminal linked the ground node.

Check the jupiter notebook "nwspikingnet_Gnd_out_of_Plate.ipynb"
'''

from matplotlib import pyplot as plt
import networkx as nx
import sys
import numpy as np
import time
# import jax.numpy as jnp
import copy
# import math
# from multiprocessing import Pool, cpu_count
import random
from scipy.sparse import csgraph, csc_matrix, csr_matrix, save_npz
from scipy import sparse
# from matplotlib import pyplot as plt
#from .mem_parameters import mem_param
from .Measure import Measure
from .ModifiedNodalAnalysis import MVNA

class create_graph():

    def define_grid_graph_2(rows, cols,
                            # src,
                            seed,
                            diag=1):
        '''
        Reference : Kevin Montano et al 2022 Neuromorph. Comput. Eng. 2 014007
        Code based on https://github.com/ MilanoGianluca/Grid-Graph_Modeling_Memristive_Nanonetworks
        :param rows:
        :param cols:
        :param seed:
        :return:
        '''

        def add_ground_node(G, GROUND_NODE_X=rows, GROUND_NODE_Y=cols, add_x=2, add_y=0):
            ground_node = (G.number_of_nodes(), {'coord': (GROUND_NODE_X + add_x, GROUND_NODE_Y + add_y)})
            # connected_to_electrode = [(node, ground_node[0]) for node, feat in G.nodes(data=True) if
            #                           node in src]
            G.add_nodes_from([ground_node])
            try:
                weights = [w['weight'] for u, v, w in G.edges(data=True)]
                w_gnd = np.max(weights)
            except:
                w_gnd = 1
            # G.add_edges_from(connected_to_electrode, weight=w_gnd)

        ##define a grid graph

        Ggrid = nx.grid_graph(dim=[rows, cols])
        if diag == 1:
            # print('Add rnd diagonals to grid-graph')
            # Define random diagonals
            if seed:
                random.seed(seed)
            else:
                random.seed(time.time())

            for r in range(rows - 1):
                for c in range(cols - 1):
                    k = random.randint(0, 1)
                    if k == 0:
                        Ggrid.add_edge((c, r), (c + 1, r + 1))
                    else:
                        Ggrid.add_edge((c + 1, r), (c, r + 1))
        ##define a graph with integer nodes and positions of a grid graph
        G = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='coord')
        add_ground_node(G=G)
        # for h in hidden_src:
        #    G.add_edge((h, G.number_of_nodes()))
        return G


class MemNet(Measure, MVNA):
    def __init__(self, mem_param, net_param, gnd, src, hidden_src, diag=1, add_opAmp=False, G_root=None):
        self.gnd = gnd
        self.src = copy.deepcopy(src)
        self.mem_param = mem_param
        self.net_param = net_param

        self.diag = diag

        if mem_param and net_param:
            self.number_of_sources = len(self.src)

            self.G_root = G_root if G_root is not None else create_graph.define_grid_graph_2(rows=net_param.rows,
                                                                                             cols=net_param.cols,
                                                                                             seed=net_param.seed,
                                                                                             diag=diag)
            # self.number_of_nodes = int(self.net_param.rows * self.net_param.cols)
            self.number_of_nodes = self.G_root.number_of_nodes()

            self.Adj = nx.to_scipy_sparse_array(self.G_root, format='csr')
            self.number_of_edges = self.Adj.data.sum() // 2
            self.adj_indexes = np.argwhere(self.Adj > 0)
            self.triangular_adj_indexes = np.argwhere(sparse.triu(self.Adj, k=0) > 0)

            if not (0 <= self.net_param.frac_of_static_elements <= 1):
                raise ValueError('Range of net_param.frac_of_static_elements should be in range [0,1].')
            self.number_of_dyn_edges = int((1 - self.net_param.frac_of_static_elements) * self.number_of_edges)
            # Random sample dynamic edges
            random.seed(net_param.mem_seed)
            self.dynamic_el_index = random.sample(range(self.number_of_edges), k=self.number_of_dyn_edges)

            # if self.number_of_dyn_edges != self.number_of_edges:

            # print('Mixed Mem-resistor:\n\t{:.1f}% memristor\n\t{:.1f}% resistor'.format(self.number_of_dyn_edges/self.number_of_edges*100,
            #                                                                  self.net_param.frac_of_static_elements*100))

            if self.number_of_dyn_edges == 0:
                self.update_edge_weights = self.resistor_edges_update
            else:
                self.update_edge_weights = self.update_DynEdge_weights

            # self.dynamic_el_index = np.arange(0, self.number_of_edges-len(src))
            # self.number_of_dyn_edges = len(self.dynamic_el_index)

            # else:
            #     self.update_edge_weights = self.update_AllEdge_weights

            # dynamic_elemts = self.triangular_adj_indexes[self.dynamic_el_index]
            self.reset()


    def node_map(self, x, y, rows, cols):
        '''

        Args:
            x: coordinate
            y: coordinate
            rows: # of rows in the net
            cols: # of rows in the net

        Returns: integer label/index of the node

        '''
        node = rows * (y + 1) - (x + 1)
        return int(node)

    def select_all_edges_except_those_linking_ground(self):
        '''
        This method defines all edges except those linked to the ground as memristive edges.
        This method is useful if the ground is on the memristive substrate.
        Returns:

        '''
        # Select all edges except those connecting to the ground
        self.dynamic_el_index = self.triangular_adj_indexes[:, 1] != gnd[0]
        self.number_of_dyn_edges = self.dynamic_el_index.sum()

    def add_opAmp(self):
        self.number_of_sources = len(self.src) + 1
        # positive, negative, output
        from collections import namedtuple
        terminals = namedtuple("Terminals", "positive negative output")
        self.opamp_terminals = terminals(9, 10, 12)

        self.src.append(self.opamp_terminals.output)
        self.gnd = [11]

        # Qui lo scalare 2 si riferisce ai nodi aggiunti non appartenent al film:
        # il ground e l'output dell'OPAmp

        self.Adj_film = self.Adj
        Adj = np.zeros((self.Adj.shape[0]+2, self.Adj.shape[1]+2))
        Adj[:-2, :-2] = self.Adj_film.todense()
        Adj[self.opamp_terminals.negative, self.opamp_terminals.output] = 1
        Adj[self.opamp_terminals.output, self.opamp_terminals.negative] = 1
        # Add ground node
        Adj[self.opamp_terminals.positive, self.gnd[0]] = 1
        Adj[self.gnd[0], self.opamp_terminals.positive] = 1

        self.Adj = csr_matrix(Adj, dtype=int)
        self.adj_indexes = np.argwhere(self.Adj > 0)
        self.triangular_adj_indexes = np.argwhere(sparse.triu(self.Adj, k=0) > 0)

        # Qui lo scalare 2 si riferisce ai nodi aggiunti non appartenent al film:
        # il ground e l'output dell'OPAmp
        self.number_of_nodes = self.G_root.number_of_nodes() + 2
        self.number_of_edges = self.Adj.data.sum() // 2

    def reset(self):
        self.G = copy.deepcopy(self.G_root)
        self.initialize_graph_attributes()
        # return 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])
        self.build_Bmat()
        self.build_Cmat()
        self.build_Dmat()
        self.incidence_mat_red = self.build_ReducedIncidenceMat()

    def change_input_src(self, new_src):
        self.src = new_src
        self.number_of_sources = len(self.src)
        self.build_Bmat()
        self.build_Cmat()
        self.build_Dmat()
        self.incidence_mat_red = self.build_ReducedIncidenceMat()

    def initialize_graph_attributes(self):
        # Initialize edge value
        if self.net_param.weight_init == 'rand':
            self.Condmat = csr_matrix((np.random.uniform(low=self.mem_param.g_min, high=self.mem_param.g_max,
                                                         size=self.number_of_edges),
                                       (self.triangular_adj_indexes.transpose()[0, :],
                                        self.triangular_adj_indexes.transpose()[1, :])),
                                      shape=self.Adj.shape)
        else:
            self.Condmat = sparse.triu(self.Adj, k=0) * self.mem_param.g0

        if self.net_param.weight_init == 'good_OC':
            # Good Ohmic resistors
            mask_static_edges = np.ones(self.number_of_edges, dtype=bool)
            mask_static_edges[self.dynamic_el_index] = False
            self.Condmat.data[mask_static_edges] = self.mem_param.g_max

        self.g = np.zeros(self.number_of_dyn_edges)

        # Node Voltage respect to ground
        self.node_Voltage = np.zeros(self.number_of_nodes)


    def compute_Imat(self):
        " Computes directed graph of currents "
        I = csr_matrix((np.multiply(self.dVmat.data, self.Condmat.data),
                        (np.array(self.triangular_adj_indexes).transpose()[0, :],
                         np.array(self.triangular_adj_indexes).transpose()[1, :])),
                       shape=self.Adj.shape)
        I = I - I.T
        return I.multiply(I > 0)

    #################  - UPDATE EDGE WEIGHT (Miranda's model)   -    ##############
    def update_AllEdge_weights(self, delta_t):

        self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * np.abs(self.dVmat.data))
        self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * np.abs(self.dVmat.data))

        # a = np.divide(self.kp, self.kp + self.kd)
        temp = np.multiply(1 + np.divide(self.kd, self.kp), self.g)
        b = np.multiply(1-temp, np.exp(- ((self.kp + self.kd) * delta_t)))
        self.g = np.multiply(np.divide(self.kp, self.kp + self.kd), 1 - b)
        self.Condmat.data = self.mem_param.g_min * (1 - self.g) + self.mem_param.g_max * self.g

        # if self.g.min()<0 or self.g.max()>1:
        #     raise ValueError('g out of interval [0,1].\n')
        # if self.Condmat.data.max() > self.mem_param.g_max or self.Condmat.data.min()< self.mem_param.g_min:
        #     raise ValueError('Conductance value out of range.\n')

    def resistor_edges_update(self, delta_t):
        pass

    def update_DynEdge_weights(self, delta_t):
        " Update of conductance of the memristor edges "
        # np.multiply can be used only on np.array (or matx.data of scipy.csr sparse matrix)!

        self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * np.abs(self.dVmat.data[self.dynamic_el_index]))
        self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * np.abs(self.dVmat.data[self.dynamic_el_index]))
        #
        # self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * self.dVmat.data[self.dynamic_el_index])
        # self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * self.dVmat.data[self.dynamic_el_index])

        ## Gaussian Noise
        # std_eps = np.multiply(1 - np.exp*(-2*(self.kp + self.kd)), np.divide(sigma**2, 2*(self.kp + self.kd)))
        # eps = np.random.normal(0, std_eps)

        # a = np.divide(self.kp, self.kp + self.kd)
        temp = np.multiply(1 + np.divide(self.kd, self.kp), self.g)
        # temp = 1 + np.multiply(np.divide(self.kd, self.kp), self.g)

        b = np.multiply(1-temp, np.exp(- ((self.kp + self.kd) * delta_t)))
        self.g = np.multiply(np.divide(self.kp, self.kp + self.kd), 1 - b)
        self.Condmat.data[self.dynamic_el_index] = self.mem_param.g_min * (1 - self.g) + self.mem_param.g_max * self.g

        # if self.g.min()<0 or self.g.max()>1:
        #     raise ValueError('g out of interval [0,1].\n')
        # if self.Condmat.data.max() > self.mem_param.g_max or self.Condmat.data.min()< self.mem_param.g_min:
        #     raise ValueError('Conductance value out of range.\n')

    def run(self, t_list, groundnode_list, sourcenode_list, V_list, save_path=None, save_mode_all=0):

        delta_t = t_list[1] - t_list[0]
        net_conductance = np.zeros(len(t_list))
        net_entropy = np.zeros(len(t_list))

        if save_path:
            print('Saving triu conductance matrix in:\n\t{:s}'.format(save_path))
            _ = self.mvna(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
            net_conductance[0] = -self.source_current[0] / V_list[0][0]
            net_entropy[0] = self.net_entropy_from_conductances(self.Condmat)
            save_npz(matrix=self.Condmat, file=save_path+'t{:09d}.npz'.format(0))
            # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(0))

            # sys.stdout.write("\r\tNetwork Stimulation: {:d}/{:d}".format(1, len(t_list)))
            if save_mode_all == 1:
                for t in range(1, len(t_list)):
                    self.update_edge_weights(delta_t=delta_t)
                    _ = self.mvna(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
                    # if (t % 2): pass
                    # else:
                    net_conductance[t] = -self.source_current[0] / V_list[0][t]
                    net_entropy[t] = self.net_entropy_from_conductances(self.Condmat)
                    # net_conductance[t] = 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])
                    save_npz(matrix=self.Condmat, file=save_path + '/t{:09d}.npz'.format(t))
                    # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(0))

            else:
                for t in range(1, len(t_list)):
                    self.update_edge_weights(delta_t=delta_t)
                    _ = self.mvna(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
                    # if (t % 2): pass
                    # else:
                    # net_conductance[t] = 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])
                    net_conductance[t] = -self.source_current[0] / V_list[0][t]
                    net_entropy[t] = self.net_entropy_from_conductances(self.Condmat)

                save_npz(matrix=self.Condmat, file=save_path+'/t{:09d}.npz'.format(t))
                # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(t))
                # sys.stdout.write("\r\nNetwork Stimulation completed\n\n")

            np.save(file=save_path + 'net_conductance.npy', arr=net_conductance)
            np.save(file=save_path + 'net_entropy.npy', arr=net_entropy)

        else:
            # Run without any saving
            H_list = [[] for t in range(len(t_list))]
            curr_rec = np.zeros((len(self.src), len(t_list)))
            # H_list[0] = self.mvna(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
            # sys.stdout.write("\r\t\tNetwork Stimulation: {:d}/{:d}".format(1, len(t_list)))
            for t in range(0, len(t_list)):
                H_list[t] = self.mvna(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
                # Current flowing into the elctrode is positive, current flowing out the electrode is negative
                curr_rec[:, t] = - self.source_current
                self.update_edge_weights(delta_t=delta_t)

            return H_list, curr_rec
        # sys.stdout.write("\r\t\tNetwork Stimulation: {:d}/{:d}".format(t+1, len(t_list)))

        # H = nx.DiGraph(H_list[0].todense())
        # # H = nx.DiGraph(I)
        # coordinates = [(node, (feat['coord'])) for node, feat in self.G.nodes(data=True)]
        # nx.set_node_attributes(H, dict(coordinates), 'coord')
        # from .visualize import visualize
        # visualize.plot_network(H, show=True, numeric_label=True)


if __name__ == "__main__":
    from easydict import EasyDict as edict

    mem_param = edict({'kp0': 2.555173332603108574e-06,  # model kp_0
                       'kd0': 6.488388862524891465e+01,  # model kd_0
                       'eta_p': 3.492155165334443012e+01,  # model eta_p
                       'eta_d': 5.590601016803570467e+00,  # model eta_d
                       'g_min': 1e-03,  # model g_min
                       'g_max': 2,  # model g_max
                       'g0': 1e-03  # model g_0
                       })

    net_param = edict({'rows': 21,
                       'cols': 21,
                       'frac_of_static_elements': 0,  # fraction of resistors in the network
                       'weight_init': None,  # 'rand',
                       'seed': 2})

    sim_param = edict({'T': 1.5,  # 4e-3, # [s]
                       'sampling_rate': 500  # [Hz]  # =steps / T  # [Hz]
                       })

    t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)  # [s]
    ########## Define source and ground pads lists ##########
    src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [390]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    labels = [(gnd[0], 'Gnd'), (src[0], 'Src')]
    ###########################################################
    #plot_dependency_from_voltage()
    #chack_conductance()

    #V_list = np.array([np.array([1]*10), np.array([5]*10)]).transpose()
    src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [390]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    #net_param.rows = 20
    #net_param.cols= 20
    V_list = np.asarray([8 if t <= 2e-3 else 0.1 for t in t_list]).transpose()
    src = [2]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [22]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    # Instantiate Classes
    net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src)
    net.run(t_list=t_list, sourcenode_list=src, groundnode_list=gnd, V_list=V_list)
