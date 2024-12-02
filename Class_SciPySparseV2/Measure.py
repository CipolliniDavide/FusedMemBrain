import networkx as nx
import numpy as np
from copy import deepcopy
# from .MemNetwork import MemNet
from scipy.sparse import csgraph, csr_matrix, csc_matrix
from scipy import sparse

class Measure():
    # def __init__(self):
        # self.Class_SciPySparseV2 = MemNet(mem_param=None, net_param=None, gnd=None, src=None).Class_SciPySparseV2


    def calculate_network_conductance(self, nodeA, nodeB, V_read=.1):
        temp_dVmat = deepcopy(self.dVmat.data)
        temp_s = deepcopy(self.source_current)
        temp_v = deepcopy(self.node_Voltage)
        _ = self.MVNA(groundnode_list=[nodeA], sourcenode_list=[nodeB], V_list=np.reshape([V_read], (1, 1)), t=0)
        net_cond = - self.source_current[0] / V_read
        self.dVmat.data = temp_dVmat
        self.source_current = temp_s
        self.node_Voltage = temp_v
        return net_cond


    def effective_resistance(self, nodeA, nodeB, sparse_matx=None):
        if sparse_matx is not None:
            return nx.resistance_distance(G=nx.Graph(sparse_matx).to_undirected(), nodeA=nodeA, nodeB=nodeB,
                                          weight='weight', invert_weight=False)
        else:
            return nx.resistance_distance(G=nx.Graph(self.Condmat).to_undirected(), nodeA=nodeA, nodeB=nodeB,
                                       weight='weight', invert_weight=False)

    # Implemented for only one source and one ground node and for the case the number of nodes is rows*cols
    def conductance_map(self, Vbias=1, trace_retrace_rate_Hz=1, sampling_rate_Hz=1e2):
        self.reset()
        delta_t = 1 / sampling_rate_Hz
        a = np.arange(self.number_of_nodes).reshape((self.net_param.rows, self.net_param.cols))
        source_loc = list(np.hstack((a, np.flip(a, axis=1))).reshape(-1))
        source_loc = list(filter(lambda a: a != self.gnd[0], source_loc))
        time_on_each_node = 1/(trace_retrace_rate_Hz * 2 * self.net_param.cols) # [Hz]
        if sampling_rate_Hz/2 < 1/time_on_each_node:
            raise ValueError('Sampling rate must be higher than sampling rate on single node.')

        condMat_list = list()
        T = time_on_each_node
        t_list = np.arange(0,  T + 1 / sampling_rate_Hz, 1 / sampling_rate_Hz)  # [s]
        current_map = np.zeros(shape=(self.net_param.rows*self.net_param.cols, 2*len(t_list)))
        count = np.zeros(self.number_of_nodes)

        for e, source in enumerate(source_loc):
            count[source] += 1
            for t in range(0, len(t_list)):
                # print(source, t)
                condMat_list.append(self.solve_MVNA_for_currents([source], self.gnd,
                                                                 np.reshape([Vbias]*len(t_list),
                                                                            newshape=(1, len(t_list))), t))
                # if count[source] < 2:
                current_map[source, int(len(t_list)*(count[source]-1) + t)] = self.matX[-1]
                self.update_edge_weights(delta_t=delta_t)

        number_of_loc = self.number_of_nodes*2 - 2
        time_tot = np.arange(0,  (T + 1 / sampling_rate_Hz)*number_of_loc, 1 / sampling_rate_Hz)  # [s]

        return - current_map, condMat_list, time_tot, source_loc

    def net_entropy_from_conductances(self, sparseMat):
        '''
        Measure entropy from conductances in the network.
        self.Condmat must be triangular!
        :return:
        '''
        CondMat_norm = sparseMat.data / sparseMat.data.sum()
        return - np.multiply(CondMat_norm.data, np.log(CondMat_norm.data)).sum()

    #######################  - CALCULATE V source     -    ########################
    # def Voutput_read(self, H_list, V_node_read):
    #     # V_out = [[] for nr in range(len(V_node_read))]
    #     V_out = np.zeros(shape=(len(V_node_read), len(H_list)))
    #     for t, H in enumerate(H_list):
    #         for v in range(len(V_node_read)):
    #             V_out[v, t] = self.calculate_Vsource(H=H, sourcenode=V_node_read[v])
    #
    #     # print('\n')
    #     # for t in range(len(H_list)):
    #     #    for nr in range(len(V_node_read)):
    #     #        V_out[nr] += [H_list[t].nodes[V_node_read[nr]]['V']]
    #     #    sys.stdout.write("\rOutput Voltage Reading: " + str(t + 1) + '/' + str(len(H_list)))
    #     return V_out

    #################  - CALCULATE  NETWORK RESISTANCE     -    ####################
    # def calculate_network_resistance(self, H, sourcenode, groundnode, V_read=1):
        #V_read = 1  # arbitrary
        # H = H.to_undirected()
        # H_pad = self.Class_SciPySparseV2(G=H, sourcenode_list=[sourcenode], groundnode_list=[groundnode], V_list=[[V_read]], t=0)
        #
        # I_fromsource = 0
        # for u, v in H_pad.edges(sourcenode):
        #     a = H_pad[u][v]['I']
        #     I_fromsource = I_fromsource + a
        #
        # Rnetwork = V_read / I_fromsource
        # return Rnetwork

    # def effective_resistance(self, nodeA, nodeB):
        #     # matL = csr_matrix(csgraph.laplacian(self.Condmat, normed=False).t)
        #     matL = csc_matrix(csgraph.laplacian(self.Condmat, normed=False))
        #     # matL_reduced = np.delete(np.delete(matL, self.gnd[0], axis=0), self.gnd[0], axis=1)
        #     Linv = sparse.linalg.inv(matL).todense()
        #     # temp = sparse.find(Linv)
        #     return Linv[nodeA, nodeA] + Linv[nodeB, nodeB] - 2*Linv[nodeA, nodeB]

    #######################  - CALCULATE I source     -    ########################
    def calculate_Isource(self, H, sourcenode):
        I_from_source = 0

        for u, v in H.edges(sourcenode):
            a = H[u][v]['I']
            I_from_source = I_from_source + a

        return I_from_source

    def Iread(self, Iread_list, H_list):
        I_output = np.zeros(shape=(len(Iread_list), len(H_list)))  # [[]] * len(Iread_list)
        for t, H in enumerate(H_list):
            for v in range(len(Iread_list)):
                I_output[v, t] = self.calculate_Isource(H=H, sourcenode=Iread_list[v])
        return I_output
