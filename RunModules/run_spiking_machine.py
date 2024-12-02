import numpy as np
from copy import deepcopy


def run_spiking_machine(net, inputsignal, hidden_src, src, gnd, id_list, neur_param, ind_time_start=0):

    def refractory_count(t, t_interval, spk_rec, mask_hidden_src):
        '''
        Make sure there was no spike in the last tp steps for the hidden neurons.
        Returns a mask that is used to update
        only neurons' voltage output of those that have not spiked in the last t_interval time steps.

        N.b. It is not exactly the a counter for the refractory period but rather a counter for the
        whole spike process: the spike and the refractory. But this also depend on what value you give to t_interval.

        :return: refractory_counter is an array where elemnt-i is True if ith-neuron didn't spike in the last tp steps, else False
        '''

        if t - t_interval < 0:
            latest_spk = spk_rec[:, :t]
        else:
            latest_spk = spk_rec[..., t - t_interval:t]
        refractory_counter[mask_hidden_src == True] = np.sum(latest_spk, axis=1) == 0
        return refractory_counter

    def update_Voltage_list(refractory_mask, rst, V_list, neur_param, t):
        # Update only neurons that did not spike in the last tp + tn steps:
        # If the neuron is in hidden list and did not spiked in the last tp steps
        # V_list[refractory_counter, t:t + neur_param.tp] = \
        #     (np.array([spike_amplitude * rst[refractory_counter[mask_hidden_src]]])).T  # + mem
        # # Negative voltage post-spike
        # V_list[refractory_counter, t + neur_param.tp - neur_param.tn:t + neur_param.tp] = \
        #     (np.array([-1 * rst[refractory_counter[mask_hidden_src]]])).T  # + mem
        #
        # Update only neurons that did not spike in the last tp + tn steps:
        # If the neuron is in hidden list and did not spike in the last tp steps
        V_list[refractory_mask, t:t + neur_param.tp] = (np.array([neur_param.spike_amplitude * rst])).T  # + mem
        # Negative voltage post-spike
        V_list[refractory_mask, t + neur_param.tp - neur_param.tn:t + neur_param.tp] = \
            (np.array([neur_param.spike_neg_amplitude * rst])).T  # + mem

    # Record voltages of nodes
    node_voltage_list = []
    # Record conductance matrix for plots
    H_list = [[] for t in range(len(inputsignal.t_list))]

    spike_amplitude = neur_param.spike_amplitude #*np.ones(neur_param.tp)
    # spike_amplitude[-len(spike_amplitude)//3:] = -1
    V_thresh = neur_param.threshold
    tau_mem = neur_param.tau_mem
    tau_Ca = neur_param.tau_Ca

    # spike_amplitude = 3
    # V_thresh = 50
    # tau_mem = 10e-3

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]
    # beta = float(np.exp(-delta_t/tau_mem))

    def spike_fn(x):
        out = np.zeros_like(x)
        out[x > 0] = 1.0
        return out

    mem = np.zeros(len(hidden_src))
    Ca = np.zeros(len(hidden_src))

    # Here we define two lists which we use to record the membrane potentials and output spikes
    mem_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    spk_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    # rst_old = np.zeros(shape=len(inputsignal.src))
    curr_rec = np.zeros(shape=((len(src), len(inputsignal.t_list))))
    Ca_rec = np.zeros(shape=((len(hidden_src), len(inputsignal.t_list))))
    mean_firing_rate_groups = np.zeros((6, len(inputsignal.t_list)-ind_time_start))

    # Mask to select currents only for the hidden units
    mask_hidden_src = np.array([True if i in hidden_src else False for i in src])
    refractory_counter = np.array([True if i in hidden_src else False for i in src])
    net.mask_hidden_src = mask_hidden_src
    # Start simulation
    print('Start simulation...')
    for t in range(ind_time_start, len(inputsignal.t_list)):
        # print(t+1, '/', len(inputsignal.t_list))

        # Check for spikes: not all of these will be counted: refractory period will select a smaller part of them
        mthr = mem - V_thresh
        rst = spike_fn(mthr)

        # Update voltage list for Hidden neurons
        refractory_mask = refractory_count(t=t, t_interval=neur_param.tp + neur_param.tn,
                                           spk_rec=spk_rec, mask_hidden_src=mask_hidden_src)
        update_Voltage_list(refractory_mask=refractory_mask, rst=rst[refractory_mask[mask_hidden_src]],
                            V_list=inputsignal.V_list, neur_param=neur_param, t=t)

        # Add noise to all input electrodes: noise is different at all time steps
        inputsignal.V_list[:, t] = inputsignal.V_list[:, t] + \
                                   inputsignal.sim_param.noise_sigma*\
                                   np.sqrt(tau_mem)*np.random.normal(0, 1, size=len(inputsignal.V_list))

        # Update network of memristors. H_list: conductance matrix for plots
        h = net.mvna(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
        net.update_edge_weights(delta_t=delta_t)
        # eff_cond.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=.1))
        H_list[t] = deepcopy(h)
        node_voltage_list.append(deepcopy(net.node_Voltage))

        # Record activity
        mem_rec[:, t] = mem
        # Place a 1 only if it did spike and if in the refractory mask there is also a True,
        # which means that the neuron is not in the refractory period
        spk_rec[:, t] = rst*refractory_mask[mask_hidden_src]
        curr_rec[:, t] = deepcopy(net.source_current)
        Ca_rec[:, t] = Ca

        # Update membrane potential:
        # Current flowing into synapses:
        # if current flows INTO is positive, if flows out of the electrode is negative
        I_ext = np.clip(deepcopy(net.source_current[mask_hidden_src]), a_min=0, a_max=1e12)

        # Either reset mem potential if spike, else update it,
        # when using the refractory mask neurons still in the refractory won't update the membrane potential
        # Update membrane potential with Ornstein-Uhlenbeck process

        # Deterministic update of the membrane potential
        mem = update_membrane_potential_deterministic(V=mem, I_ext=I_ext, delta_t=delta_t,
                                                      tau_mem=tau_mem, C_m=neur_param.C_m,
                                                      rst=rst, refractory_m=refractory_mask[mask_hidden_src])
        # Calcium current
        Ca = Ca * float(np.exp(-delta_t / tau_Ca)) + rst*refractory_mask[mask_hidden_src]

        ####
        if id_list:
            for i, lab in enumerate(np.unique(id_list)):
                mean_firing_rate_groups[i] = spk_rec[id_list == lab, t].sum()

    return mem_rec, curr_rec, spk_rec, H_list, node_voltage_list, Ca_rec, mean_firing_rate_groups


def update_membrane_potential_deterministic(V, I_ext, delta_t, tau_mem, rst, refractory_m, C_m=1):
    temp = float(np.exp(-delta_t/tau_mem)) * V + (I_ext/C_m)*delta_t  # + noise_term
    # new_mem = (1.0 - rst)*(curr)
    new_V = temp * (1.0 - rst) * refractory_m
    return new_V

def update_membrane_potential_ou(V, I_ext, delta_t, tau_mem, sigma, rst, refractory_m, C_m=1, V_rest=0):

    """
    Simulate the Ornstein-Uhlenbeck process with spiking behavior and refractory period

    # Parameters
      V_rest = -65.0           # Resting potential in mV
      V_thresh = -50.0         # Spiking threshold in mV
      V_reset = -70.0          # Reset potential after a spike in mV
      tau_m = 20.0             # Membrane time constant in ms
      sigma = 5.0              # Noise strength
      C_m = 1.0                # Membrane capacitance in uF/cm^2
    """

    # Compute potential change with OU process and external input
    dW = np.random.normal(0, np.sqrt(delta_t))  # Wiener process
    dV = (V_rest - V) / tau_mem * delta_t + (I_ext / C_m) * delta_t + sigma * dW
    return V + dV * (1.0 - rst) * refractory_m
