import copy
from easydict import EasyDict as edict
import numpy as np
import os
import glob
import networkx as nx
from networkx.algorithms.bipartite.basic import color
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import truncnorm, alpha
import pickle

from Class_SciPySparseV2.anim_vis import plot_H_evolution
from Class_SciPySparseV2.MemNet import MemNet
from Class_SciPySparseV2.visualize import visualize
from Class_SciPySparseV2 import visual_utils


# from helpers_wires.helpers_plot2d import plot_nw_net, plot_nw_net_graph

def plot_graph_grid(H_list, t_list, node_labels, pos, n=5, step=50, min_G=None, max_G=None,
                    save_fold=None, fig_name=None, V_list=None):
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3))  # Create n x n grid
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    if (min_G is not None) and (max_G is not None):
        norm = mcolors.LogNorm(vmin=min_G, vmax=max_G)  # Log scale normalization
    else:
        weights_all = [w for H in H_list for w in
                       nx.get_edge_attributes(nx.from_scipy_sparse_array(H), 'weight').values()]
        norm = mcolors.Normalize(vmin=min(weights_all), vmax=max(weights_all))  # Normalize over all weights

    cmap = plt.cm.Reds
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)  # For colorbar

    for i, ind in enumerate(np.arange(0, len(H_list), step)[:n * n]):
        ax = axes[i]  # Select subplot
        G_temp = nx.from_scipy_sparse_array(H_list[ind], create_using=nx.Graph())

        # Extract edge weights for coloring
        edges, weights = zip(*nx.get_edge_attributes(G_temp, 'weight').items())

        # Plot the network
        ax.set_title(f"t={t_list[ind]:.3f}")
        nx.draw(G_temp, labels=node_labels, edge_color=[cmap(norm(w)) for w in weights],
                width=2, pos=pos, node_size=2, ax=ax)

    # Hide empty subplots if H_list is smaller than n*n
    for j in range(i + 1, n * n):
        fig.delaxes(axes[j])

    # Add color bar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Position for colorbar
    fig.colorbar(sm, cax=cbar_ax, label='Edge Weight')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    if save_fold and fig_name:
        plt.savefig(f"{save_fold}/{fig_name}_griddynamics.png", dpi=300)
    plt.show()


def identify_farthest_nodes(pos):
    nodes = list(pos.keys())
    positions = np.array([pos[n] for n in nodes])

    # Compute all pairwise distances
    dist_matrix = squareform(pdist(positions, metric='euclidean'))

    # Find the indices of the max distance
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    # Get the actual node labels
    node1, node2 = nodes[i], nodes[j]
    max_distance = dist_matrix[i, j]
    return node1, node2, max_distance


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def truncated_normal(mean, std, size, left_lim=0., right_lim=np.inf):
    return truncnorm.rvs(a=(left_lim - mean) / std, b=(right_lim - mean) / std, loc=mean, scale=std, size=size)


net_param = edict({'rows': 21,  # 31,
                   'cols': 21,  # 31,
                   'frac_of_static_elements': 0,  # fraction of resistors in the network
                   'weight_init': None,  # rand, if rand start with random, else: None
                   'weight_seed': None,  # weight_seed,
                   'diag_seed': 2,
                   'diag_flag': 1
                   })

net_param_static = edict({'rows': 21,  # 31,
                   'cols': 21,  # 31,
                   'frac_of_static_elements': .9999999,  # fraction of resistors in the network
                   'weight_init': None,  # rand, if rand start with random, else: None
                   'weight_seed': None,  # weight_seed,
                   'diag_seed': 2,
                   'diag_flag': 1
                   })

sim_param = edict({'T':300,  # 4e-3, # [s]
                   'sampling_rate': 100  # [Hz]  # =steps / T  # [Hz]
                   })

def compute_unitary_edgeconductance_at_infty(kp0, kd0, eta_p, eta_d, V_read):
    """ V read must be > 0 """
    g = kp0*np.exp(eta_p*np.abs(V_read)) / (kp0*np.exp(eta_p*np.abs(V_read)) + kd0*np.exp(-eta_d*np.abs(V_read)))
    return g

mem_param = edict({'kp0': 0.02445256479617862,  # model kp_0
                   'kd0':  0.053463576839283544,  # model kd_0 # kd0 modulates the volatility of the memristor
                   'eta_p': 0.6093328954928459,  # model eta_p
                   'eta_d': 3.3475141532239103,  # model eta_d
            # as t → \infty: the normalized conductance g = kp(V=0.01) / (kp(V=0.01) + kd(V=0.01)) = 0.32,
                   # The (real) conductance value for each edge will therefore be: G = g_min + (g_max - g_min) * g,
                   # hence, unless g = 0, the real conductance G does not converge to G_min.

                   'g_min': 0.0006427599616447571,  # model g_min
                   'g_max': 0.0020643151904970974,  # model g_max
                   'g0' : .0006
                   })

time_array = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)  # [s]
V_read = .01
V_max = 2.0
pulse = V_max * np.ones_like(time_array)

mem_param_static = edict({'kp0': 0.02445256479617862,  # model kp_0
                   'kd0':  0.053463576839283544,  # model kd_0 # kd0 modulates the volatility of the memristor
                   'eta_p': 0.6093328954928459,  # model eta_p
                   'eta_d': 3.3475141532239103,  # model eta_d
                          # as t → ∞: the normalized conductance g = kp(V=0.01) / (kp(V=0.01) + kd(V=0.01)) = 0.32,
                          # The (real) conductance value for each edge will therefore be: G = g_min + (g_max - g_min) * g,
                          # hence, unless g = 0, the real conductance G does not converge to G_min.
                   'g_min': 0.00064,  # model g_min
                   'g_max': 0.0020643151904970974,  # model g_max
                   # 'g0': 0.0006  # model g_0
                  'g0': 0.00064 + (0.002 - 0.00064) * compute_unitary_edgeconductance_at_infty(kp0=0.02445256479617862,
                                                                                               kd0=0.053463576839283544,
                                                                                               eta_p=0.6093328954928459,
                                                                                               eta_d=3.3475141532239103,
                                                                                               V_read=V_read),
                   })


if __name__ == "__main__":
    fig_format="png"
    save_fold_fig = "./"
    ####################################### Topology ###################################################################
    # Graph topology: grid network with random diagonals
    from Class_SciPySparseV2.MemNet import create_graph
    G = create_graph.define_grid_graph_2(rows=net_param['rows'], cols=net_param['cols'],
                                         diag_flag=net_param['diag_flag'], diag_seed=net_param['diag_seed'])
    src = [31]
    gnd = [409]
    electrodes_node_labels = [(gnd[0], 'Gnd')] + [(s, 'Src') for s in src]

    # Graph topology: heterogeneous sample
    file = 'sample_it=000000_Nnw=001000.pickle'
    G = nx.convert_node_labels_to_integers(pickle_load(file), first_label=0)
    # Get positions and select the couple of nodes that are the farthest
    pos = nx.get_node_attributes(G, 'pos')
    node1, node2, max_distance = identify_farthest_nodes(pos)
    src = [node1]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [node2]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    electrodes_node_labels = [(gnd[0], 'Gnd')] + [(s, 'Src') for s in src]

    # Get positions and select the couple of nodes that are the farthest
    node_labels = {node: label for node, label in electrodes_node_labels}
    pos = nx.get_node_attributes(G, 'pos')

    pulse[time_array>100] = V_read
    pulse[:20] = V_read
    end_pulse_time = 2e-1
    print("Reading voltage:", np.min(pulse))
    signal_name = 'pulse'
    V_list = np.array([pulse])
    plt.plot(time_array, pulse, label=signal_name)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage Appl. [V]")
    plt.show()

    # Name fig based on parameters
    fig_name = f"{signal_name:s}_Vmax={np.max(V_list[0]):07.3f}"
    ############################################# Dynamics #############################################################
    # mem_param.eta_p = truncated_normal(mean=64, std=2000, size=G.number_of_edges(), left_lim=1e-15, right_lim=1e15)
    # plt.hist(mem_param.kd0)
    # plt.show()

    # instantiate the network class
    net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src, G_root=G)
    # Instantiate Classes
    H_list, curr_rec, node_voltages = net.run(t_list=time_array, sourcenode_list=src, groundnode_list=gnd, V_list=V_list)

    # Static network
    net_static = MemNet(mem_param=mem_param_static, net_param=net_param_static, gnd=gnd, src=src, G_root=G)
    H_list_static, curr_rec_static, node_voltages_static = net_static.run(t_list=time_array,
                                                                          sourcenode_list=src,
                                                                          groundnode_list=gnd,
                                                                          V_list=V_list)

    fig, ax1 = plt.subplots(sharex=True, nrows=1, figsize=(6, 4), layout='tight')
    ax1.plot(time_array, curr_rec[0], '--', label=r'$I$', color='green')
    ax1.set_ylabel(r'$I$' + ' [A]', color='green')
    plt.show()

    conductance = curr_rec[0] / V_list[0]
    fig, (ax1, ax2) = plt.subplots(sharex=True, nrows=2, figsize=(6, 6), layout='tight')
    ax1.plot(time_array, curr_rec[0] / V_list[0], '--', label=r'$G_{src-gnd}$', color='green')
    ax1.plot(time_array, curr_rec_static[0] / V_list[0], '--', label=r'$G^{static}_{src-gnd}$', color='blue')

    # ax1.axhline(y=C, ls='--', color='red')
    ax1.legend(loc='best')
    ax1.set_ylabel(r'$G_{src-gnd}$' + ' [S]', color='green')

    # ax_twinx = ax2.twinx()
    # ax_twinx.plot(time_array, curr_rec[0], c='red')
    # ax_twinx.set_ylabel(r'Current [A]', color='red')
    ax2.plot(time_array, V_list[0], color='blue')
    ax2.set_ylabel(r'V [V]', color='blue')
    ax2.set_xlabel('Time [s]')
    # plt.savefig(f"{./dynamics_{fig_name}_dynamics.{fig_format}", dpi=300)
    plt.show()
    plt.close()


    ##############################################################################################################
    conductances = np.array([h.data for h in H_list])


    max_G = np.max(np.array([h.data for h in H_list]))
    min_G = np.min(np.array([h.data for h in H_list]))
    G_avg = np.array([h.data.mean() for h in H_list])
    G_std = np.array([h.data.std() for h in H_list])



    # # Plot
    # fig, axes = plt.subplots(nrows=3, figsize=(8,6), layout='tight')
    # axes[0].plot(time_array, conductances.mean(1))
    # # axes[1].plot(time_array, curr_rec[0])
    # for v in node_voltages[20:30]:
    #     axes[1].plot(time_array, v/V_list[0], alpha=0.5)
    # axes[2].plot(time_array, V_list[0])
    # axes[1].set_ylabel(r'$V_i/V_{app}$', color='blue', fontsize=20)
    # axes[2].set_ylabel(r'$V_{app}$'+ ' [V]', color='blue', fontsize=20)
    # axes[2].set_xlabel('Time [s]', fontsize=20)
    # axes[0].set_ylabel(r'$G_{avg}$' + ' [S]', color='black', fontsize=20)
    # plt.show()

    # Plot network in time
    plot_graph_grid(H_list, time_array, node_labels, pos, n=4, step=50,
                    save_fold=save_fold_fig, fig_name=fig_name)

    # visualize.plot_network(G=net.G, numeric_label=True, labels=node_labels, figsize=(8, 8), node_size=10)
    # plt.show()
    ####################################################################

    # fig, (ax1, ax2) = plt.subplots(sharex=True, nrows=2, figsize=(6, 6), layout='tight')
    # ax1.plot(time_array, G_avg, label=r'$G_{avg}$', color='black')
    # ax1.fill_between(time_array, G_avg - G_std, G_avg + G_std, color='grey', alpha=0.3)
    # ax1_tw = ax1.twinx()
    # ax1_tw.plot(time_array, curr_rec[0] / V_list[0], '--', label=r'$G_{src-gnd}$', color='green')
    # ax1.set_ylim(bottom=min_G - min_G / 100, top=max_G + max_G / 100)
    # # ax1.legend(loc='best')
    # ax1.set_ylabel(r'$G_{avg}$' + ' [S]')
    # ax1_tw.set_ylabel(r'$G_{src-gnd}$' + ' [S]', color='green')
    #
    # ax_twinx = ax2.twinx()
    # ax_twinx.plot(time_array, curr_rec[0], c='red')
    # ax_twinx.set_ylabel(r'Current [A]', color='red')
    # ax2.plot(time_array, V_list[0], color='blue')
    # ax2.set_ylabel(r'V [V]', color='blue')
    # ax2.set_xlabel('Time [s]')
    # # plt.savefig(f"{./dynamics_{fig_name}_dynamics.{fig_format}", dpi=300)
    # plt.show()
    # plt.close()



