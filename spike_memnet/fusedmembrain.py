import os
from os.path import join
import argparse
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import json

from Class_SciPySparseV2.visual_utils import set_ticks_label, create_colorbar
from Class_SciPySparseV2.anim_vis import plot_H_evolution
from Class_SciPySparseV2.make_gif import make_gif
from Class_SciPySparseV2.MemNet_gndOut import MemNet
from Class_SciPySparseV2.ControlSignal import ControlSignal
from Class_SciPySparseV2.visualize import visualize
from Class_SciPySparseV2.utils import utils

from RunModules.Reservoir_wth_floating_gate import node_map, generate_random_points
from helpers_dataset.convert_format import convert_format_of_dataset
from helpers_dataset.load_dataset import data_loader

from helpers_plot_function.spiking_help_plot import (raster_plot, plot_hidneur_mem, pca_of_dynamics,
                                                     plot_neuron_average_spike_rate, plot_ntw_average_firing_rate)
from RunModules.run_spiking_machine import run_spiking_machine


def save_data(save_path):
    np.save(arr=mem_rec, file=save_path + 'mem_rec.npy')
    np.save(arr=curr_rec, file=save_path + 'curr_rec.npy')
    np.save(arr=Ca_rec, file=save_path + 'Ca_rec.npy')
    np.save(arr=spk_rec, file=save_path + 'spk_rec.npy')
    np.save(arr=node_voltage_list, file=save_path + 'node_voltage_list.npy')
    np.save(arr=hidden_src, file=save_path + 'hidden_src.npy')
    np.save(arr=src, file=save_path + 'src.npy')
    for d, name in zip([sim_param, neur_param, mem_param, net_param],
                       ['sim_param', 'neur_param', 'mem_param', 'net_param']):
        np.save(file=save_path + 'name.npy', arr=d)
        utils.save_dict_to_txt(dict=d, savepath=save_path, name=name)
    # Load
    # read_dictionary = np.load(save_data_sim + 'sim_param.npy', allow_pickle='TRUE').item()


def load_data(load_path):
    # Load numpy arrays
    mem_rec = np.load(load_path + 'mem_rec.npy')
    curr_rec = np.load(load_path + 'curr_rec.npy')
    Ca_rec = np.load(load_path + 'Ca_rec.npy')
    spk_rec = np.load(load_path + 'spk_rec.npy')
    node_voltage_list = np.load(load_path + 'node_voltage_list.npy')
    hidden_src = np.load(load_path + 'hidden_src.npy')
    src = np.load(load_path + 'src.npy')

    # Load dictionaries
    # sim_param = np.load(load_path + 'sim_param.npy', allow_pickle=True).item()
    # neur_param = np.load(load_path + 'neur_param.npy', allow_pickle=True).item()
    # mem_param = np.load(load_path + 'mem_param.npy', allow_pickle=True).item()
    # net_param = np.load(load_path + 'net_param.npy', allow_pickle=True).item()

    # If the parameters were also saved as text, load them using the custom method
    sim_param_txt = utils.load_dict_from_txt(load_path, 'sim_param')
    neur_param_txt = utils.load_dict_from_txt(load_path, 'neur_param')
    mem_param_txt = utils.load_dict_from_txt(load_path, 'mem_param')
    net_param_txt = utils.load_dict_from_txt(load_path, 'net_param')

    return (mem_rec, curr_rec, Ca_rec, spk_rec, node_voltage_list, hidden_src, src,
            # sim_param, neur_param, mem_param, net_param,
            sim_param_txt, neur_param_txt, mem_param_txt, net_param_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='OUTPUT_Spk/PLOTOutputRandDiag_GndOut', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=41, type=int)
    # parser.add_argument('-crt_sim_data', '--create_sim_data', default=0, type=int)
    parser.add_argument('-b_start', '--batch_start', default=0, type=int)
    parser.add_argument('-b_end', '--batch_end', default=20, type=int)
    parser.add_argument('-p', '--frac_of_mem', default=1, type=float, help='Fraction of network memristive edges.')
    parser.add_argument('-el_seed', '--electrodes_seed', default=1, type=int,
                        help="Seed for output electrodes location")
    parser.add_argument('--ds_seed', default=1, type=int, help="Seed for dataset shuffling, thus selection of samples.")
    parser.add_argument('--diag_seed', default=1, type=int, help="Seed for the random diagonals in the network.")
    parser.add_argument('--mem_seed', default=1, type=int, help="Seed for the assignation of the memristors edges in the network.")
    parser.add_argument('-diag', '--random_diagonals', default=1, type=int,
                        help="Flag to have or not the random diagonals in the square lattice.")
    parser.add_argument('-img_form', '--img_format', default='pdf', type=str)
    parser.add_argument('-v', "--verbose", action='store_false')
    args = parser.parse_args()


    ############################## Params #################################################

    # Use 1 second (aka T= 10e-1) for the panel a and its bottom inset (the waves),
    # for the top inset change to T=.1e-1

    sim_param = edict({'T': 10.e-1,  # [s]
                       # 'steps': 100,
                       'sampling_rate': 1e4,  # [Hz]  # =steps / T  # [Hz]
                       # dt = T / steps  # [s] or    dt = 1/sampling_rate bc  steps= sampling_rate *T
                       'noise_sigma': 0,
                       'Vbias': 1.5})


    # Fraction of static elements [0, 1] is used to set the fraction of Ohmic edges in the network
    net_param = edict({'rows': args.linear_size,
                       'cols': args.linear_size,
                       'frac_of_static_elements': 1 - args.frac_of_mem,
                       'weight_init': None,  # 'rand',
                       'seed': args.diag_seed,
                       'mem_seed': args.mem_seed,
                       'random_diagonals': 1})

    # Threshold needs to be of the same order of magnitude of g0, g_min and g_max.
    neur_param = edict({'spike_amplitude': 1.2,  # [V]
                        'spike_neg_amplitude': -.1,  # [V]
                        'threshold': 5e-1,
                        # 'threshold': np.random.normal(0, 1, size=20),
                        'C_m': 3.5e-20 * sim_param.sampling_rate,  # Farads [F]
                        'tp': 5, # number of time steps
                        'tn': 3, # number of time steps
                        'tau_Ca': sim_param.T / 10,
                        'tau_mem': 1e-3, # seconds
                        'noise_sigma': 0})

    mem_param = edict({'kp0': 2.56e-06,  # model kp_0
                       'kd0': 64.9,  # model kd_0 # kd0 modulates the volatility of the memristor
                       'eta_p': 34.90,  # model eta_p
                       'eta_d': 5.59,  # model eta_d
                       'g_min': 1e-12,  # 1e-12,  # model g_min # [Ohm-1]
                       'g_max': 200e-12,  # 200e-12,  # model g_max [Ohm-1]
                       'g0': 1e-12  # model g_0
                       })

    # Prepare save folder
    root = os.getcwd()
    save_path_sim = join(root,
                         '{:}/SavePlotEvolN{:03d}/{:}_Vbias{:.1f}_Vspk{:.1f}_{:.1f}_Threshold{:.2f}_GmaxGmin{:.0e}/tp_f{:.0f}_tn{:.0f}_NoiseInp{:.2f}_NoiseNeur{:.2f}/'.format(
                             args.save_path, net_param.rows,
                             net_param.weight_init, sim_param.Vbias,
                             neur_param.spike_amplitude,
                             neur_param.spike_neg_amplitude,
                             neur_param.threshold,
                             mem_param.g_max / mem_param.g_min,
                             neur_param.tp,
                             neur_param.tn,
                             sim_param.noise_sigma,
                             neur_param.noise_sigma))
    utils.ensure_dir(save_path_sim)
    save_data_sim = save_path_sim + '/DataSim/'
    utils.ensure_dir(save_data_sim)

    ############################################################################################
    # Generic input

    # Set Input Nodes, Hidden Nodes and Ground nodes
    # input_src = [0, 60, 258, 352]
    input_src = [0]
    np.seed = 1
    # hidden_src = sorted(list(np.random.choice(range(1, args.linear_size ** 2), 100, replace=False)))
    hidden_src = sorted([746, 1230, 1261, 1145, 1425, 681, 1511, 1537, 710, 1488, 1223, 1417, 339, 332, 254, 218, 371, 1236, 1656, 1471, 1258, 1159, 1375, 939, 445, 929, 818, 227, 620, 773, 744, 1553, 1367, 1606, 462, 1383, 1429, 41, 98, 720, 1155, 1071, 1525, 394, 743, 1342, 12, 586, 1593, 140, 362, 847, 179, 192, 1441, 7, 1197, 567, 828, 237, 830, 1378, 1005, 777, 431, 1468, 798, 855, 104, 1102, 443, 913, 363, 1190, 1655, 1115, 460, 1268, 1539, 534, 686, 761, 571, 870, 1210, 444, 1380, 1350, 208, 247, 1141, 1569, 940, 361, 4, 1641, 953, 1456, 1029, 1374])

    # hidden_src = sorted([150, 138, 334, 65, 349, 276, 17, 55, 269, 296, 191, 36, 361, 196, 189, 5, 218, 109, 363, 85, 311, 23])
    # hidden_src = [2, 7]
    src = sorted(input_src + hidden_src)
    gnd = [args.linear_size**2]
    input_source_labels = [(input_src[i], 'Inp{:d}'.format(input_src[i])) for i in range(len(input_src))]
    hidden_source_labels = [(hidden_src[i], 'n{:d}'.format(hidden_src[i])) for i in range(len(hidden_src))]
    node_labels = [(gnd[0], 'Gnd')] + input_source_labels + hidden_source_labels


    ####################################### Reservoir input ############################################################
    ####################################################################################################################
    # Convert the dataset format in suc a way that each node is mapped to a location in the network.
    # We define an inner-square, which is the portion of 2d space where input electrodes are located.
    # We define an inner-square, which is the portion of 2d space where input electrodes are located.
    # The output nodes (`hidden_nodes`) will be located on the outer-square.

    # classes_list = [0]
    # classes_str = '_'.join(map(str, classes_list))
    #
    # train_set = data_loader(dataset_name='mnist',
    #                         data_root_fold=f"{os.path.abspath(os.curdir).rsplit('/', 1)[0]}",
    #                         seed=args.ds_seed,
    #                         show_info=False,
    #                         num_samp=10,
    #                         classes_list=classes_list,
    #                         train=True)
    # # test_set = data_loader(dataset_name='mnist', seed=args.ds_seed, classes_list=cl_list, train=False, num_samp=5)
    #
    # X, Y, coord_electrodes = convert_format_of_dataset(dataset_lst=[train_set],
    #                                                    net_param=net_param,
    #                                                    n_input=20,
    #                                                    linear_fraction_of_inner_square=7,
    #                                                    show_info=False)
    # # Select only one sample for this logbook
    # # X = [X[0]]
    # # Y = [Y[0]]
    # # coord_electrodes = [coord_electrodes[0]]
    #
    # # Create the dataset to be fed to the reservoir class.
    # dataset = edict({'X': X, 'Y': Y, 'coord_inp_electrodes': coord_electrodes})
    # # num = 0000
    # # plt.scatter(coord_electrodes[num][:, 0], coord_electrodes[num][:, 1], c=dataset.X[num])
    # # plt.title(Y[num])
    # # plt.show()
    #
    # # Create hidden nodes (i.e., output electrodes) and displace them in the outer square.
    # # Note that this is related to the size of inner square that is used to place input electrodes when
    # # the dataset is created.
    # random_points = generate_random_points(N=60, outer_square_linsize=net_param.rows,
    #                                        inner_square_linsize=(net_param.rows) / 2 + 4,  #
    #                                        seed=args.electrodes_seed)
    # hidden_src = sorted([node_map(x, y, rows=net_param.rows, cols=net_param.cols) for x, y in random_points])
    #
    # idx=0
    # pos_electrodes = dataset.coord_inp_electrodes[idx]
    # intensity = dataset.X[idx]
    # label = dataset.Y[idx]
    #
    # # Set input electrodes
    # input_src = [node_map(x, y, rows=net_param.rows, cols=net_param.cols) for x, y in pos_electrodes]
    # src = sorted(input_src + hidden_src)

    ####################################################################################################################
    ####################################### Instantiate memristor network class ########################################

    net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src, hidden_src=hidden_src, diag=args.random_diagonals)
    coordinates = [(node, (feat['coord'])) for node, feat in net.G.nodes(data=True)]
    # visualize.plot_network(G=net.G, numeric_label=True, labels=node_labels, figsize=(8,8), node_size=10)

    # Plot distribution of electrodes
    color_node_vis = np.zeros(net.number_of_nodes)
    color_node_vis[input_src] = 10
    color_node_vis[hidden_src] = 5
    visualize.plot_network(G=net.G, numeric_label=False,
                           # labels=node_labels,
                           figsize=(8, 8),
                           node_size=80,
                           node_shape='s',
                           node_color=color_node_vis,
                           )
    plt.savefig("./electrodes_distribution.pdf", dpi=300)
    # plt.show()
    if args.verbose:
        plt.show()
    else: plt.close()


    # Input signal class
    inputsignal = ControlSignal(sources=src, sim_param=sim_param)
    for inp in input_src:
        # inputsignal.V_list[src.index(inp)] = np.ones(inputsignal.t_list.shape[0]) * sim_param.Vbias
        kick = 10
        inputsignal.V_list[src.index(inp), 0:kick] = np.ones(kick) * sim_param.Vbias
    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]

    # Start Simulation
    mem_rec, curr_rec, spk_rec, H_list, node_voltage_list, Ca_rec, _ = run_spiking_machine(net=net,
                                                                                           inputsignal=inputsignal,
                                                                                           hidden_src=hidden_src,
                                                                                           src=src,
                                                                                           gnd=gnd,
                                                                                           neur_param=neur_param,
                                                                                           id_list=None,
                                                                                           ind_time_start=0)
    save_data(save_path=save_data_sim)
    print('Sim. concluded')

    # # Load data
    # (mem_rec, curr_rec, Ca_rec, spk_rec, node_voltage_list,
    #  hidden_src, src, sim_param, neur_param, mem_param, net_param) = load_data(save_data_sim)

    ####################################################################################################################
                                                # Plots
    ####################################################################################################################


    raster_plot(t_list=inputsignal.t_list, spk_rec=spk_rec, save_path_sim=save_path_sim, img_format=args.img_format, show=args.verbose)
    # pca_of_dynamics(X=Ca_rec[:, :], save_path=save_path_sim, name='Ca', img_format=args.img_format, )
    # pca_of_dynamics_gif(X=Ca_rec[:, :], t_list=inputsignal.t_list, save_path=save_path_sim, name='Ca', figsize=(8, 6),
    #                     ratio=20)

    plot_hidneur_mem(inputsignal=inputsignal, time=inputsignal.t_list/1e-3,
                     mem_rec=mem_rec, spk_rec=spk_rec, neur_param=neur_param,
                     node_labels=node_labels,
                     img_format=args.img_format, src=src, hidden_src=hidden_src, save_path_sim=save_path_sim,
                     node_index=0,
                     x_label=r'$\mathbf{Time~(ms)}$',
                     figname='spike_and_V_sign')

    # for i in range(1, curr_rec.shape[0]//5):
    i=0
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    ax[0].plot(inputsignal.t_list, curr_rec[net.mask_hidden_src][i])
    ax[1].plot(inputsignal.t_list, mem_rec[i])
    ax[2].eventplot(spk_rec[i] * inputsignal.t_list)
    ax[3].plot(inputsignal.t_list, inputsignal.V_list[net.mask_hidden_src][i])
    ax[1].axhline(y=neur_param.threshold, linestyle='--', alpha=.7, c='red')
    for ax, lab in zip(ax, [r'$\mathbf{I_{ext}} (A)$', r'$\mathbf{V_m} (V)$', 'Raster', 'Appl. Volt (V)']):
        ax.set_xlim((0, .02))
        ax.set_ylabel(lab)
    set_ticks_label(ax=ax, ax_type='x', data=inputsignal.t_list, num=3, ax_label=r'$\mathbf{Time (s)}$')
    plt.tight_layout()
    plt.savefig(save_path_sim+'curr_rast_mem.png')
    if args.verbose:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    time = inputsignal.t_list / 1e-3
    for i in range(inputsignal.V_list.shape[0]):
        ax[0].plot(time, inputsignal.V_list[i])
        ax[1].plot(time, inputsignal.V_list[i])
    for i in range(2):
        set_ticks_label(ax=ax[i], data=inputsignal.V_list.reshape(-1), num=7, ticks=[0, neur_param.spike_amplitude,
                                                                                     neur_param.spike_neg_amplitude,
                                                                                     sim_param.Vbias],
                    ax_type='y', ax_label='Volt. Hid-neur.\n(V)', valfmt='{x:.1f}')
    x_lim = (10, 20)
    ax[1].set_xlim(x_lim)
    set_ticks_label(ax=ax[1], data=time, num=4, ax_type='x', ax_label=r'$\mathbf{Time (ms)}$')
    set_ticks_label(ax=ax[0], data=time, num=4, ax_type='x', ax_label='')
    plt.tight_layout()
    plt.savefig(save_path_sim + 'V_sign.{:s}'.format(args.img_format))
    if args.verbose:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(len(hidden_source_labels)):
        ax.plot(inputsignal.t_list, Ca_rec[i])
    set_ticks_label(ax=ax, data=inputsignal.t_list, num=7, ax_type='x', ax_label='Time (s)')
    set_ticks_label(ax=ax, data=Ca_rec, num=7, ax_type='y', ax_label='Ca', valfmt='{x:.1f}')
    plt.tight_layout()
    plt.savefig(save_path_sim + 'Ca_traces.{:s}'.format(args.img_format))
    if args.verbose:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(inputsignal.t_list, Ca_rec.mean(axis=0))
    set_ticks_label(ax=ax, data=inputsignal.t_list, num=7, ax_type='x', ax_label='Time (s)')
    set_ticks_label(ax=ax, data=Ca_rec.mean(axis=0), num=7, ax_type='y', ax_label='<Ca>')
    plt.tight_layout()
    if args.verbose:
        plt.show()
    else:
        plt.close()

    # plot_neuron_average_spike_rate(spk_rec[:20], time=inputsignal.t_list, save_path_sim=save_path_sim,
    #                                 img_format=args.img_format, verbose=args.verbose,
    #                                window_size=.2)
    # plt.show()

    plot_ntw_average_firing_rate(spk_rec, time=inputsignal.t_list, save_path_sim=save_path_sim,
                                    img_format=args.img_format, window_size=10, y_label=r"$\mathbf{Fr_{ntw}~(Hz)}$")

    # corr_matx = np.mean(np.einsum('...i,...j', mean_activity.T, mean_activity.T), axis=0)
    corr_matx = np.corrcoef(spk_rec)
    corr_matx[range(len(corr_matx)), range(len(corr_matx))] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(corr_matx)
    create_colorbar(fig=fig, ax=ax, mapp=im, array_of_values=corr_matx[~np.isnan(corr_matx)], valfmt="{x:.2f}",
                    fontdict_cbar_label={'label': 'Correlation'},
                    fontdict_cbar_tickslabel=None, fontdict_cbar_ticks=None)
    ax.set_title('Mean corr: {:.3f}'.format(np.nanmean(corr_matx)), weight='bold', size='xx-large')
    plt.tight_layout()
    plt.savefig(save_path_sim+'correlation.{:s}'.format(args.img_format))
    if args.verbose:
        plt.show()
    else:
        plt.close()

    # plot_neuron_activity(inputsignal=inputsignal, mem_rec=mem_rec,
    #                      hidden_source_labels=hidden_source_labels,
    #                      input_source_labels=input_source_labels,
    #                      threshold=1,
    #                      save_path=save_path_sim,
    #                      figname='neur_activity.{:s}'.format(args.img_format))
    #
    # plot_neuron_activity(inputsignal=inputsignal, mem_rec=curr_rec,
    #                      hidden_source_labels=input_source_labels+hidden_source_labels,
    #                      input_source_labels=input_source_labels,
    #                      threshold=1,
    #                      save_path=save_path_sim,
    #                      figname='current_flowing_into_electrodes.{:s}'.format(args.img_format))


    mean_g = np.array([np.mean(c_mat.data) for c_mat in H_list]) / 1e-12
    std_g = np.array([np.std(c_mat.data) for c_mat in H_list]) / 1e-12

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(inputsignal.t_list, mean_g)
    ax.fill_between(inputsignal.t_list, mean_g - std_g, mean_g + std_g, alpha=0.2, linewidth=4)
    set_ticks_label(ax=ax, ax_type='x', data=inputsignal.t_list, num=4, ax_label='Time (s)')
    set_ticks_label(ax=ax, ax_type='y', data=[mean_g + std_g, mean_g], num=4, ax_label=r'$\mathbf{\bar{G}}$ (pS)')
    plt.tight_layout()
    plt.savefig(save_path_sim + f"network_conductance.{args.img_format}", dpi=300)
    # if args.verbose:
    plt.show()
    # else:
    #     plt.close()

    index_hidden_src_to_plot = 0
    for node_size in [90]:
    # node_size = 100
        plot_H_evolution(H_Horizon=H_list,
                         t_list_Horizon=inputsignal.t_list,
                         numeric_label=False,
                         show_name_labels=False,
                         # signals_2nd_row=[mem_rec[index_hidden_src_to_plot]],
                         desired_eff_conductance=None,
                         # signals_1rst_row=inputsignal.V_list,
                         node_labels=node_labels,
                         y_label_2ndrow='$\mathbf{MemPot}$'+'_{:s}'.format(hidden_source_labels[index_hidden_src_to_plot][1]),
                         src=src,
                         coordinates=coordinates,
                         number_of_plots=len(H_list)//50,
                         node_voltage_list=node_voltage_list,
                         # title='p={:.2f}, '.format(1 - net_param.frac_of_static_elements) + \
                         #        r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(mem_param.g_max/mem_param.g_min),
                         save_path=save_path_sim+'/Nodesize{:d}'.format(node_size),
                         figsize=(14,14),
                         node_size=node_size,
                         edge_width=8,
                         image_format="png", #args.img_format,
                         cmap = plt.cm.cividis_r,
                         edges_cmap=plt.cm.Reds,
                         inpnode_c="green",
                         activenode_c="mediumorchid",
                         # inpnode_c="green",
                         # activenode_c="darkviolet",
                         node_shape="s",
                         activenode_shape="s",
                         inputnode_shape="d",
                         node_alpha=.99,
                         background_color='lavenderblush'
                         )

        make_gif(frame_folder=save_path_sim+'/Nodesize{:d}/H_Evolution/'.format(node_size), gif_name="my_awesome_nodesize{:d}".format(node_size),
                 images_format=args.img_format, save_path=save_path_sim, duration=200)

    a = 0
    # def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5):
    #     gs=GridSpec(*dim)
    #     if spk is not None:
    #         dat = 1.0*mem
    #         dat[spk>0.0] = spike_height
    #         dat = dat.detach().cpu().numpy()
    #     else:
    #         dat = mem.detach().cpu().numpy()
    #     for i in range(np.prod(dim)):
    #         if i==0: a0=ax=plt.subplot(gs[i])
    #         else: ax=plt.subplot(gs[i],sharey=a0)
    #         ax.plot(dat[i])
    #         ax.axis("off")


    # fig=plt.figure(dpi=100)
    # plot_voltage_traces(mem_rec, spk_rec)