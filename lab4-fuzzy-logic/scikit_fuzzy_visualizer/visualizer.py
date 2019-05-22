import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from skfuzzy import interp_membership
from skfuzzy.control import ControlSystemSimulation, Antecedent, Consequent
from skfuzzy.control.controlsystem import CrispValueCalculator
from skfuzzy.control.fuzzyvariable import FuzzyVariable


class FuzzyVariablesVisualizer:

    def __init__(self, sim: ControlSystemSimulation, interval: int = 100, style: str = 'ggplot'):
        """ Creates visualizer """
        self.sim = sim

        # get variables
        variables = list(sim.ctrl.fuzzy_variables)

        # init matplotlib
        plt.style.use(style)
        self.fig, axs = plt.subplots(len(variables), gridspec_kw={'hspace': 1})
        self.fig.canvas.set_window_title('Fuzzy variables visualization')

        # create plots
        self.plots = [FuzzyVariablePlot(var, ax, self.sim) for var, ax in zip(variables, axs)]

        # create animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update,
            init_func=self._init_plot, blit=False, interval=interval
        )

    def run(self):
        """ Starts visualizing """
        plt.show()

    def _init_plot(self):
        """ Init function for plot animation, (can be called multiple times!!!) """
        for p in self.plots:
            p.init_plot()

    def _update(self, iter):
        """ Update function for plot animation, """
        for p in self.plots:
            p.update()


class FuzzyVariablePlot:

    def __init__(self, var: FuzzyVariable, ax: Axes, sim: ControlSystemSimulation):
        self.var = var
        self.ax = ax
        self.sim = sim

        self.plots = {}
        self.cut_plots = []

        # create plots
        for label, term in self.var.terms.items():
            self.plots[label] = self.ax.plot(self.var.universe, term.mf, label=label)[0]

        # create crisp
        self.crisp = self.ax.plot([0,0], [0, 0], color='k', lw=3, label='crisp value')[0]

    def init_plot(self):
        """ Plot initialization (can be called multiple times!!!) """

        # axes
        self.ax.set_ylim(0, 1.01)
        self.ax.set_xlim(self.var.universe.min(), self.var.universe.max())

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.ax.get_xaxis().tick_bottom()
        self.ax.get_yaxis().tick_left()

        self.ax.tick_params(direction='out')

        self.ax.set_title(
            self.var.label,
            fontweight='bold' if isinstance(self.var, Consequent) else 'normal'
        )

        # legend
        self.ax.legend(framealpha=0.5, loc='right')

    def update(self):
        """ Updates plot values """

        # calculate values
        crispy = CrispValueCalculator(self.var, self.sim)
        ups_universe, output_mf, cut_mfs = crispy.find_memberships()

        # remove old cut plots
        for cut_plot in self.cut_plots:
            cut_plot.remove()
        del self.cut_plots[:]

        # create cut plots
        for label, plot in self.plots.items():
            if label in cut_mfs:
                self.cut_plots.append(self.ax.fill_between(ups_universe, cut_mfs[label], facecolor=plot.get_color(), alpha=0.4))

        # if crisp value not available
        if len(cut_mfs) == 0 or all(output_mf == 0) or not self._get_crisp():
            self.crisp.set_data([0, 0], [0, 0])

        # if crisp value available
        else:
            cv = self._get_crisp()
            y = max(
                interp_membership(self.var.universe, term.mf, cv)
                for key, term in self.var.terms.items()
                if key in cut_mfs
            )

            # small cut values are hard to see, so simply set them to 1
            if y < 0.1:
                y = 1.

            # plot it
            self.crisp.set_data([cv, cv], [0, y])

    def _get_crisp(self):
        if isinstance(self.var, Antecedent):
            return self.var.input[self.sim]
        elif isinstance(self.var, Consequent):
            return self.var.output[self.sim]
