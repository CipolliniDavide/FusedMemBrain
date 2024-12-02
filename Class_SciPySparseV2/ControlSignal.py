import scipy.signal as signal
import random
import numpy as np

class ControlSignal():

    def __init__(self, sources, sim_param):
        self.sim_param = sim_param
        self.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)  # [s]
        self.sim_param.dt = self.t_list[1] - self.t_list[0]
        self.src = sources
        self.V_list = np.zeros(shape=(len(self.src), len(self.t_list)))


    def square_ramp(self, StartAmplitude=1, number_of_cycles=5):
        signal_freq = number_of_cycles / self.sim_param.T
        s = StartAmplitude / 2 + StartAmplitude / 2 * signal.square(2 * np.pi * signal_freq * self.t_list)
        count = (s.sum() - StartAmplitude)/StartAmplitude/number_of_cycles * 2
        r = np.repeat(np.arange(1, number_of_cycles + 1), count)
        sig = s * np.hstack((r, [0]))
        sig[sig == 0] = .1
        return np.roll(sig, shift=1)

    def linear_ramp(self, Vmax, start=.1):
        return np.linspace(start=start, stop=Vmax, num=len(self.t_list))

    def plot_V(self, ax=None, labels=None):
        from matplotlib import pyplot as plt
        from .visual_utils import set_ticks_label, set_legend

        if ax is None:
            fig = plt.figure('volt_input', figsize=(10, 10))
            ax = fig.add_subplot(111)
        # ax.set_title('Voltage Input', fontsize=25)
        for v in range(len(self.V_list)):
            float_index = [i for i in range(len(self.V_list[v])) if self.V_list[v][i] == 'f']
            value_index = [i for i in range(len(self.V_list[v])) if self.V_list[v][i] != 'f']
            if labels==None:
                p = ax.plot([self.t_list[i] for i in value_index], [float(self.V_list[v][i]) for i in value_index],
                            label='Node ' + str(self.src[v]), linewidth=2)
            else:
                p = ax.plot([self.t_list[i] for i in value_index], [float(self.V_list[v][i]) for i in value_index],
                            label=labels[v], linewidth=2)
            color = p[0].get_color()
            ax.plot([self.t_list[i] for i in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        set_legend(ax=ax, title='', ncol=1, loc=0)
        set_ticks_label(ax=ax, ax_type='y', data=self.V_list, ax_label='Voltage [V]')
        set_ticks_label(ax=ax, ax_type='x', data=self.t_list, ax_label='Time [s]')
        ax.grid()
        return ax


def prepare_signal(A1, signal_freq, t_list):
    sign = A1/2 + A1/2*signal.square(2 * np.pi * signal_freq * t_list)
    return sign
