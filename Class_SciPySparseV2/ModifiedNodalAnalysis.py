from scipy.sparse import csgraph, csc_matrix, csr_matrix, save_npz
from scipy import sparse
import numpy as np
from matplotlib import pyplot as plt

class MVNA():

    def __init__(self):
        pass

    def build_Bmat(self):
        B = np.zeros(shape=(self.number_of_nodes, self.number_of_sources), dtype=int)
        for j, k in zip(self.src, range(len(self.src))):
            B[j, k] = 1
        B = np.delete(B, self.gnd[0], axis=0)
        self.B = csr_matrix(B)

    def build_Cmat(self):

        if hasattr(self, 'opamp_terminals'):
            C = np.zeros(shape=(self.number_of_sources, self.number_of_nodes), dtype=int)
            for j, k in zip(self.src[:-1], range(len(self.src)-1)):
                C[k, j] = 1
            C = np.delete(C, self.gnd[0], axis=1)
            # OpAmp
            C[-1, self.opamp_terminals.positive] = 1
            C[-1, self.opamp_terminals.negative] = -1
        else:
            C = self.B.transpose()
        # plt.imshow(C)
        # plt.show()
        self.C = csr_matrix(C)

    def build_Dmat(self):
        self.D = csr_matrix((self.number_of_sources, self.number_of_sources))

    def build_ReducedIncidenceMat(self):
        '''
        :return: Reduced incidence matrix
        '''
        from networkx import incidence_matrix
        return self.delete_from_csr(incidence_matrix(self.G, oriented=True).tocsr(), row_indices=[self.gnd[0]])



    def delete_from_csr(self, mat, row_indices=[], col_indices=[]):
        """
        Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
        WARNING: Indices of altered axes are reset in the returned matrix
        """
        if not isinstance(mat, csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")

        rows = []
        cols = []
        if row_indices:
            rows = list(row_indices)
        if col_indices:
            cols = list(col_indices)

        if len(rows) > 0 and len(cols) > 0:
            row_mask = np.ones(mat.shape[0], dtype=bool)
            row_mask[rows] = False
            col_mask = np.ones(mat.shape[1], dtype=bool)
            col_mask[cols] = False
            return mat[row_mask][:, col_mask]
        elif len(rows) > 0:
            mask = np.ones(mat.shape[0], dtype=bool)
            mask[rows] = False
            return mat[mask]
        elif len(cols) > 0:
            mask = np.ones(mat.shape[1], dtype=bool)
            mask[cols] = False
            return mat[:, mask]
        else:
            return mat

    def mvna(self, sourcenode_list, groundnode_list, V_list, t):
        '''
        Solve for voltages on each node and currents on each edge

        :param sourcenode_list:
        :param groundnode_list:
        :param V_list:
        :param t:
        :return:
        '''

        def build_z(t):
            # fill Z
            I = np.zeros(shape=(self.number_of_nodes-1, 1))
            e = np.zeros(shape=(self.number_of_sources, 1))
            for i in range(V_list.shape[0]):
                e[i] = V_list[i, t]
            return csr_matrix(np.vstack((I, e)))

        supply_list = np.array(sourcenode_list + groundnode_list[1:])
        num_supply = len(supply_list)

        # Definition of matrices
        matL = csr_matrix(csgraph.laplacian(self.Condmat + self.Condmat.T, normed=False))
        matL_reduced = self.delete_from_csr(mat=matL, row_indices=[groundnode_list[0]], col_indices=[groundnode_list[0]])


        # Compose matrix
        submat1 = sparse.hstack((matL_reduced, self.B))
        submat2 = sparse.hstack((self.C, self.D))
        matY = sparse.vstack((submat1, submat2), format='csr')
        matZ = build_z(t=t)
        matX = sparse.linalg.spsolve(matY, matZ)

        # add voltage (respect to gnd) as a node attribute
        mask = np.ones(self.number_of_nodes, bool)
        mask[groundnode_list[0]] = False
        self.node_Voltage[mask] = matX[:self.number_of_nodes-1].reshape(-1)
        self.source_current = matX[self.number_of_nodes-1:]

        # Branch voltage (more or less)
        dv = np.zeros(shape=int(self.number_of_edges))
        for i, ind in enumerate(self.triangular_adj_indexes):
            dv[i] = self.node_Voltage[ind[0]] - self.node_Voltage[ind[1]]

        self.dVmat = csr_matrix((dv,
                                 (self.triangular_adj_indexes.transpose()[0, :],
                                  self.triangular_adj_indexes.transpose()[1, :])),
                                shape=self.Adj.shape)

        # Branch voltage (number of el. must be equal to number of edges)
        # dv = self.incidence_mat_red.transpose().dot(np.expand_dims(self.node_Voltage[mask], axis=1))
        # self.dVmat = csr_matrix((dv.reshape(-1),
        #                          (self.triangular_adj_indexes.transpose()[0, :],
        #                           self.triangular_adj_indexes.transpose()[1, :])),
        #                         shape=self.Adj.shape)

        # Return matrix of conductance with directed edges according to current flow
        # return (self.Condmat + self.Condmat.T).multiply(self.incidence_mat_red)

        return (self.Condmat + self.Condmat.T) #.multiply(self.dVmat - self.dVmat.T > 0)


# def solve_MVNA_for_currents(self, sourcenode_list, groundnode_list, V_list, t):
    #     supply_list = np.array(sourcenode_list + groundnode_list[1:])
    #     num_supply = len(supply_list)
    #
    #     # Definition of matrices
    #     matL = csr_matrix(csgraph.laplacian(self.Condmat + self.Condmat.T, normed=False))
    #     matL_reduced = self.delete_from_csr(mat=matL, row_indices=[groundnode_list[0]],
    #                                         col_indices=[groundnode_list[0]])
    #     matD = csr_matrix((num_supply, num_supply))
    #     # fill Z
    #     temp = [i + self.number_of_nodes - 1 for i, v in enumerate(supply_list) if v in sourcenode_list]
    #     matZ = csr_matrix((V_list[:, t], (temp, [0] * len(sourcenode_list))),
    #                       shape=(self.number_of_nodes - 1 + num_supply, 1))
    #
    #     # Compose matrix
    #     submat1 = sparse.hstack((matL_reduced, matB))
    #     submat2 = sparse.hstack((matB.transpose(), matD))
    #     matY = sparse.vstack((submat1, submat2), format='csr')
    #
    #     self.matX = sparse.linalg.spsolve(matY, matZ)
    #
    #     # add voltage (respect to gnd) as a node attribute
    #     mask = np.ones(self.number_of_nodes, bool)
    #     mask[groundnode_list[0]] = False
    #     self.node_Voltage[mask] = self.matX[:self.number_of_nodes - 1].reshape(-1)
    #     # Qui da ottimizzare
    #     # with Pool(processes=cpu_count()) as pool:
    #     #     a = pool.map(self.f_deltaV, self.adj_indexes)
    #
    #     dv = np.zeros(shape=int(self.number_of_edges))
    #     for i, ind in enumerate(self.triangular_adj_indexes):
    #         dv[i] = self.node_Voltage[ind[0]] - self.node_Voltage[ind[1]]
    #
    #     self.dVmat = csr_matrix((dv,
    #                              (self.triangular_adj_indexes.transpose()[0, :],
    #                               self.triangular_adj_indexes.transpose()[1, :])),
    #                             shape=self.Adj.shape)
    #
    #     return (self.Condmat + self.Condmat.T).multiply(self.dVmat - self.dVmat.T > 0)
