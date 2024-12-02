import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
from Class_SciPySparseV2.utils import utils
from Class_SciPySparseV2.visual_utils import create_colorbar, create_discrete_colorbar, set_legend, set_ticks_label


def plot_hidneur_mem(spk_rec, mem_rec, save_path_sim, inputsignal, neur_param, src, hidden_src,
                     time,
                     node_labels,
                     node_index = 5,
                     t_sel = (0, 240), img_format='png', show=False,
                     x_label=r'$\mathbf{Time~(s)}$',
                     figname='spike_and_V_sign'):
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # for i in range(inputsignal.V_list.shape[0]):
    ax[0].plot(time[t_sel[0]:t_sel[1]], inputsignal.V_list[src.index(hidden_src[node_index]), t_sel[0]:t_sel[1]])
    for l in time[spk_rec[node_index] == 1]:
        if (l > time[t_sel[0]]) and (l < time[t_sel[1]]):
            ax[0].axvline(x=l, linestyle='--', alpha=.7, c='orange')
            ax[1].axvline(x=l, linestyle='--', alpha=.7, c='orange')
    ax[1].axhline(y=neur_param.threshold, linestyle='--', alpha=.7, c='red')
    ax[1].plot(time[t_sel[0]:t_sel[1]], mem_rec[node_index, t_sel[0]:t_sel[1]])  # , label=node_labels[i])
    ax[2].eventplot(spk_rec[node_index][t_sel[0]:t_sel[1]] * time[t_sel[0]:t_sel[1]], label=node_labels[node_index])
    # ax[1].set_xlim(0.2, 0.21)
    # events = ax[0].eventplot(spk_rec * time)

    # set_legend(ax=ax[0])

    set_ticks_label(ax=ax[2], data=time[t_sel[0]:t_sel[1]], num=4, ax_type='x', ax_label=x_label)
    set_ticks_label(ax=ax[0], data=inputsignal.V_list[node_index, t_sel[0]:t_sel[1]], num=3, ax_type='y',
                    ax_label='Voltage output')
    set_ticks_label(ax=ax[1], data=mem_rec[node_index, t_sel[0]:t_sel[1]], num=5, ax_type='y', ax_label=r'$\mathbf{V_{m}}$')
    set_ticks_label(ax=ax[2], data=[0], num=1, ax_type='y', ax_label='Raster plot', only_ticks=True)
    plt.tight_layout()
    plt.savefig(save_path_sim + '{:s}.{:s}'.format(figname, img_format))
    if show:
        plt.show()
    else:
        plt.close()


def raster_plot(t_list, spk_rec, img_format='png', save_path_sim='./', name='', lim=(0.2, .21), show=False):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[1].set_xlim(lim[0], lim[1])

    # events = ax[1].eventplot(spk_rec * t_list, colors="black", alpha=0.2)
    # events = ax[0].eventplot(spk_rec * t_list, colors="black", alpha=0.2)

    ax[0].imshow(spk_rec, aspect='auto', cmap='binary', origin='lower',
               extent=[t_list[0], t_list[-1], 0, spk_rec.shape[0]])
    ax[1].imshow(spk_rec, aspect='auto', cmap='binary', origin='lower',
                 extent=[t_list[0], t_list[-1], 0, spk_rec.shape[0]])

    set_ticks_label(ax=ax[0], ax_type='x', data=t_list, num=5, ax_label='')
    set_ticks_label(ax=ax[1], ax_type='x', data=[lim[0], lim[1]], num=3, ax_label='Time (s)')
    for i in range(len(ax)):
        set_ticks_label(ax=ax[i], ax_type='y', data=[0], ticks=np.linspace(0, len(spk_rec)-1, 7), valfmt="{x:.0f}", num=5,
                        ax_label='Neuron index')
    plt.savefig(save_path_sim + 'raster_{:s}.{:s}'.format(name, img_format))
    if show:
        plt.show()
    else:
        plt.close()


def pca_of_dynamics(X, save_path, img_format='png', name='', show=False):
    # Principal component analysys on time evolution
    # from mpl_toolkits import mplot3d
    # from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X.T)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 'blue')
    set_ticks_label(ax=ax, ax_type='x', data=X_pca[:, 0], ax_label='PC1', num=3,
                    fontdict_ticks_label={'fontweight': 8, 'fontsize': 10})
    set_ticks_label(ax=ax, ax_type='y', data=X_pca[:, 1], ax_label='PC2', num=3,
                    fontdict_ticks_label={'fontweight': 8, 'fontsize': 10})
    set_ticks_label(ax=ax, ax_type='z', data=X_pca[:, 2], ax_label='PC3', num=3,
                    fontdict_ticks_label={'fontweight': 10, 'fontsize': 10})
    plt.tight_layout()
    plt.savefig(save_path + 'pcaTime_{:s}.{:s}'.format(name, img_format))
    if show:
        plt.show()
    else:
        plt.close()


def pca_of_dynamics_gif(X, t_list, spk_rec, save_path, img_format='png', name='', figsize=(12, 6), ratio=20):
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio
    # Principal component analysys on time evolution
    from mpl_toolkits import mplot3d
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X.T)
    # Create a list to store each frame of the GIF
    frames = []

    # Iterate over each x-value and create a frame for each value
    for t in range(0, len(X_pca), ratio):
        fig = plt.figure(figsize=figsize)

        # Add the 3D plot as the first subfigure
        ax = fig.add_subplot(211, projection='3d')
        ax.plot3D(X_pca[:t, 0], X_pca[:t, 1], X_pca[:t, 2], 'blue')
        # ax = plt.axes(projection='3d')
        # ax.plot3D(X_pca[:t, 0], X_pca[:t, 1], X_pca[:t, 2], 'blue')
        set_ticks_label(ax=ax, ax_type='x', data=X_pca[:, 0], ax_label='PC1', num=3,
                        fontdict_ticks_label={'fontweight': 8, 'fontsize': 10})
        set_ticks_label(ax=ax, ax_type='y', data=X_pca[:, 1], ax_label='PC2', num=3,
                        fontdict_ticks_label={'fontweight': 8, 'fontsize': 10})
        set_ticks_label(ax=ax, ax_type='z', data=X_pca[:, 2], ax_label='PC3', num=3,
                        fontdict_ticks_label={'fontweight': 10, 'fontsize': 10})

        # Add the raster plot as the second subfigure
        ax2 = fig.add_subplot(212)
        ax2.axvline(x=t_list[t], color="red", linestyle="--", alpha=.8)
        # events = ax2.eventplot(spk_rec * t_list)
        ax2.imshow(spk_rec, aspect='auto', cmap='binary', origin='lower',
                   extent=[t_list[0], t_list[-1], 0, spk_rec.shape[0]])
        set_ticks_label(ax=ax2, ax_type='x', data=t_list, num=5, ax_label='Time [s]')
        set_ticks_label(ax=ax2, ax_type='y', data=[0], ticks=np.linspace(0, len(spk_rec) - 1, 7),
                            valfmt="{x:.0f}", num=5, ax_label='# hid. neur.')

        plt.tight_layout()
        # Save the current figure as an image and append it to the frames list
        plt.savefig(save_path+'temp_frame.{:s}'.format(img_format))
        frames.append(imageio.imread(save_path+'temp_frame.{:s}'.format(img_format)))
        plt.close()

        # Remove the temporary frame image
        os.remove(save_path+'temp_frame.{:s}'.format(img_format))
    # Save the frames as a GIF file
    imageio.mimsave(save_path+'pca_animation.gif', frames, duration=1.7)


# Plot membrane potentials and Voltage Inputs
def plot_neuron_activity(inputsignal, mem_rec, input_source_labels, hidden_source_labels,  threshold,
                         save_path, figname='neur_activity', img_format='png', show=False):
    fig = plt.figure('', figsize=(10, 10))
    ax = fig.add_subplot(212)
    # ax.set_ylim((-.5, 1.5))
    inputsignal.plot_V(ax=ax, labels=[s[1] for s in input_source_labels + hidden_source_labels])

    ax1 = fig.add_subplot(211)
    # colors = ['orange', 'green']
    # for i in range(len(hidden_src)):
    for i in np.random.choice(range(0, len(hidden_source_labels)), 5, replace=False):
        ax1.plot(inputsignal.t_list, mem_rec[i], label=hidden_source_labels[i][1], linewidth=2,
                 # color=colors[i]
                 )
    ax1.axhline(y=threshold, color="purple", linestyle="--", label='Threshold')
    # ax1.set_xlabel('Time [s]', fontsize=20)
    ax1.set_ylabel('Membrane Pot. / Currents [a.u.]', fontsize=20)
    ax1.tick_params(axis='both', labelsize='x-large')
    # ax.set_yticks(fontsize=15)
    ax1.legend(fontsize=15)
    ax1.grid()
    plt.savefig(save_path+figname + '.{:s}'.format(img_format))
    if show:
        plt.show()
    else:
        plt.close()



import scipy.stats as stats

def plot_ntw_average_firing_rate(spk_rec, time, save_path_sim, window_size=30, y_label='Average Spike Rate (Hz)',
                                 img_format='png', confidence_level=0.95, verbose=True):

    # fr = spk_rec.sum(axis=1)/time[-1]
    # plt.hist(fr)
    # plt.show()

    # Check the shape of spike recordings
    if spk_rec.ndim != 2:
        raise ValueError("spk_rec should be a 2D array with shape (n_neurons, n_timepoints)")

    dt = time[1] - time[0]

    # Calculate the window length in terms of indices (assuming uniform time steps)
    # window_length = int(window_size / dt)

    # Convolve each neuron's spike array with a moving window and normalize by window size
    firing_rate_over_time = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size), mode='same') / (dt*window_size),
        axis=1,
        arr=spk_rec
    )
    time_smoothed = np.convolve(time, np.ones(window_size), mode='same') / window_size

    # Plot the smoothed mean firing rate
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(time_smoothed, firing_rate_over_time.mean(axis=0), linewidth=4, label='Mean Firing Rate', alpha=.2)

    # for fr in firing_rate_over_time[:2]:
    #     ax.plot(time_smoothed, fr, linewidth=4, label='Mean Firing Rate', alpha=.2)

    # # Fill the area between the confidence intervals
    # ax.fill_between(time_smoothed,
    #                 smoothed_ci_lower,
    #                 smoothed_ci_upper,
    #                 color='gray', alpha=0.5, label='Confidence Interval')

    # Set labels
    set_ticks_label(ax=ax, data=firing_rate_over_time, num=7, ax_type='y', ax_label=y_label)
    set_ticks_label(ax=ax, data=time_smoothed, num=7, ax_type='x', ax_label='Time [s]')

    # Add legend
    ax.legend()

    # Save or show the plot
    plt.savefig(save_path_sim + f"smoothed_mean_firing_rate_with_ci.{img_format}", dpi=300)
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_neuron_average_spike_rate(spk_rec, time, save_path_sim, window_size = 0.2, y_label='Firing Rate (spikes/s)',
                                   img_format='png', verbose=True):
    window_size = .001
    # Define the sliding window size in seconds
    window_length = int(window_size / (time[1] - time[0]))

    # Calculate firing rate over time using convolution
    firing_rate_over_time = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_length), mode='same') / window_size,
        axis=1,
        arr=spk_rec
    )

    # Plot the firing rate for each neuron
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for neuron_idx in range(firing_rate_over_time.shape[0]):
        ax.plot(time, firing_rate_over_time[neuron_idx], label=f'Neuron {neuron_idx + 1}')

    # Set labels
    set_ticks_label(ax=ax, data=firing_rate_over_time, axis=0, num=3, ax_type='y', ax_label=y_label)
    set_ticks_label(ax=ax, data=time, num=4, ax_type='x', ax_label='Time (s)')

    # Add a legend indicating the number of neurons
    # ax.legend([f'Neuron {i+1}' for i in range(len(avg_spike_rates))], loc='upper right', fontsize='small')

    # Save or show the plot
    plt.savefig(save_path_sim + f"neuron_average_spike_rate.{img_format}", dpi=300)
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        plt.close()


