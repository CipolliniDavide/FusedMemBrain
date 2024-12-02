import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def run_oscillatory_machine(net, inputsignal, hidden_src, src, gnd, id_list, osc_param, ind_time_start=0,
                            verbose=1
                            ):
    '''

    Args:
        net:
        inputsignal:
        hidden_src:
        src:
        gnd:
        id_list:
        osc_param:
        ind_time_start:
        verbose:

    Returns:
    # solve ODEs using simple IMEX scheme
    # for gnn in self.GNNs:
    #     Y = Y + self.dt * (torch.relu(gnn(X, edge_index)) - self.alpha * Y - self.gamma * X)
    #     X = X + self.dt * Y
    # pass
    '''
    def transformation_memnet_coupling(X, t):
        inputsignal.V_list[mask_hidden_src, t] = X
        # Update network of memristors
        # H_list: conductance matrix for plots
        h = net.mvna(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
        net.update_edge_weights(delta_t=delta_t)
        # eff_cond.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=.1))
        H_list[t] = deepcopy(h)
        node_voltage_list.append(deepcopy(net.node_Voltage))

        # if current flows INTO is positive, if flows out of the electrode is negative
        # curr = np.clip(deepcopy(net.source_current[mask_hidden_src]), a_min=0, a_max=1e12)
        curr = deepcopy(net.source_current[mask_hidden_src])
        return curr

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]

    # Mask to select currents only for the hidden units
    hidden_src = hidden_src
    mask_hidden_src = np.isin(src, hidden_src)
    # input_src = np.sort(np.setxor1d(src, hidden_src))
    # input_src = np.array(src)[~mask_hidden_src]

    # Record voltages of nodes
    node_voltage_list = []
    # Record conductance matrix for plots
    H_list = [[] for t in range(len(inputsignal.t_list))]
    X_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    Y_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    hid_curr_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    inp_curr_rec = np.zeros(shape=((len(src)-len(hidden_src), len(inputsignal.t_list))))

    # Set initial values of ODEs
    # X0 = np.zeros(len(hidden_src))
    # Y0 = np.zeros(len(hidden_src))
    # X0 = 2*np.random.normal(0, 1, size=len(hidden_src))
    # Y0 = 2*np.random.normal(0, 1, size=len(hidden_src))
    X0 = np.arange(len(hidden_src))/len(hidden_src)
    Y0= np.arange(len(hidden_src))/len(hidden_src)

    X = X0
    Y = Y0

    # Start simulation
    for t in range(ind_time_start, len(inputsignal.t_list)):
        # Solve ODEs of oscillators using simple IMEX scheme
        hid_curr = transformation_memnet_coupling(X=osc_param.gain * X, t=t)
        Y = Y + delta_t * ( hid_curr - osc_param.alpha * Y - osc_param.gamma * X) # The
        X = X + delta_t * Y
        X_rec[:, t] = X
        Y_rec[:, t] = Y
        hid_curr_rec[:, t] = hid_curr
        inp_curr_rec[:, t] = deepcopy(net.source_current[~mask_hidden_src])

    return X_rec, Y_rec, hid_curr_rec, inp_curr_rec, H_list, node_voltage_list


def run_oscillatory_machine_wth_floating_gate(net,
                                              inputsignal,
                                              hidden_src,
                                              src,
                                              gnd,
                                              id_list,
                                              osc_param,
                                              ind_time_start=0, verbose=1):
    '''

    Args:
        net:
        inputsignal:
        hidden_src:
        src:
        gnd:
        id_list:
        osc_param:
        ind_time_start:
        verbose:

    Returns:
    # solve ODEs using simple IMEX scheme
    # for gnn in self.GNNs:
    #     Y = Y + self.dt * (torch.relu(gnn(X, edge_index)) - self.alpha * Y - self.gamma * X)
    #     X = X + self.dt * Y
    # pass
    '''
    def transformation_memnet_coupling(X, t):
        inputsignal.V_list[mask_hidden_src, t] = X
        # Update network of memristors
        # H_list: conductance matrix for plots
        h = net.mvna(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
        net.update_edge_weights(delta_t=delta_t)
        # eff_cond.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=.1))
        H_list[t] = deepcopy(h)
        node_voltage_list.append(deepcopy(net.node_Voltage))

        # if current flows INTO is positive, if flows out of the electrode is negative
        # curr = np.clip(deepcopy(net.source_current[mask_hidden_src]), a_min=0, a_max=1e12)
        curr = deepcopy(net.source_current[mask_hidden_src])
        return curr

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]

    # Mask to select currents only for the hidden units
    hidden_src = hidden_src
    mask_hidden_src = np.isin(src, hidden_src)
    # input_src = np.sort(np.setxor1d(src, hidden_src))
    # input_src = np.array(src)[~mask_hidden_src]

    # Record voltages of nodes
    node_voltage_list = []
    # Record conductance matrix for plots
    H_list = [[] for t in range(len(inputsignal.t_list))]
    X_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    Y_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    hid_curr_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    inp_curr_rec = np.zeros(shape=((len(src)-len(hidden_src), len(inputsignal.t_list))))

    # Set initial values of ODEs
    # X0 = np.zeros(len(hidden_src))
    # Y0 = np.zeros(len(hidden_src))
    # X0 = 2*np.random.normal(0, 1, size=len(hidden_src))
    # Y0 = 2*np.random.normal(0, 1, size=len(hidden_src))
    X0 = np.arange(len(hidden_src))/len(hidden_src)
    Y0= np.arange(len(hidden_src))/len(hidden_src)

    X = X0
    Y = Y0

    # TODO : if the signal starts with zero there are issues with the following lines detecting the input signal end
    # Time for floating gate on input
    mean_signal_input = inputsignal.V_list[~mask_hidden_src].sum(axis=0)
    index_of_end_of_input_signal = np.where(mean_signal_input == 0)[0][0]
    # plt.plot(inputsignal.V_list[~mask_hidden_src].T)
    # plt.show()

    # Start simulation
    for t in range(ind_time_start, len(inputsignal.t_list)):

        # Set floating gate on source input nodes: remove input sources from the set of electrodes
        if index_of_end_of_input_signal == t:
            inputsignal.V_list = inputsignal.V_list[mask_hidden_src]
            net.change_input_src(new_src=hidden_src)
            src = hidden_src
            mask_hidden_src = np.ones_like(hidden_src, dtype=bool)

        # Solve ODEs of oscillators using simple IMEX scheme
        hid_curr = transformation_memnet_coupling(X=osc_param.gain * X, t=t)
        Y = Y + delta_t * (hid_curr - osc_param.alpha * Y - osc_param.gamma * X)
        X = X + delta_t * Y
        X_rec[:, t] = X
        Y_rec[:, t] = Y
        hid_curr_rec[:, t] = hid_curr

        if t >= index_of_end_of_input_signal:
            inp_curr_rec[:, t] = np.full(len(inp_curr_rec), np.nan)
        else:
            inp_curr_rec[:, t] = deepcopy(net.source_current[~mask_hidden_src])

    return X_rec, Y_rec, hid_curr_rec, inp_curr_rec, H_list, node_voltage_list

