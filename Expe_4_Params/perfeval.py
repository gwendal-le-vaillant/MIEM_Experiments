
import math
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Color Maps

import experimentdataprocessing as edp
import figurefiles


class EvalType(IntEnum):
    INGAME = 0  # the in-game implemented score function (scores display for gamification feedback)
    SUM_BASED = 1
    ADJUSTED_NORM2 = 2
    ADJUSTED = 3  # Very close to the in-game, but simpler and gives a zero score to possible random answers.
    FOCUS_ON_TIME = 4
    FOCUS_ON_ERROR = 5
    COUNT = 6


def get_error_type_for_adjustment(adjustment_type):
    """ Returns the norm (1 or 2) to be used with the corresponding adjusted performance evaluation function.

    Norm-1 is generally preferred because controlled parameters produced very independant results ; it seems
     more reasonable to simply add all the errors and normalize the result,
     rather than taking the root mean square of all errors.
     """
    if adjustment_type == 2:
        return 2
    else:
        return 1


def adjusted_eval(e, t, allowed_time, adjustment_type=EvalType.ADJUSTED):
    """ Computes adjusted performance values ; the type of adjustement (int) can be chosen, including the
    norm to be used for error computation. """

    # The ZERO adjusted eval is actually the in-game perf eval.
    if adjustment_type == EvalType.INGAME:
        return expe4_ingame_eval(e, t, allowed_time)

    # Too
    elif adjustment_type == EvalType.SUM_BASED:
        precision_term = 1.0 - 2.0 * e  # null or negative if total error > 0.5
        time_term = 1.0 - t / allowed_time  # always positive, and is at least 1/3
        final_score = 0.6 * precision_term + 0.4 * time_term
        return np.clip(final_score, 0.0, 1.0)

    # not used anymore....
    elif adjustment_type == EvalType.ADJUSTED_NORM2:
        return adjusted_eval(e, t, allowed_time, adjustment_type=EvalType.ADJUSTED)

    # - - - ADJUSTED PERFORMANCE EVALUATION - - -
    # This function is not exactly the same as the in-game perf eval function, but it does not change
    # the significance of results. Synths 3 to 8 gives clearly better results the interp, and synth 0, 1, 2
    # and 9 still give similar results for both interp/sliders method.
    # It does not change the polynomial fits of perf vs. expertise data.
    # ---> Advantages/disadvantages (compared to in-game perf eval function):
    # + removes possible random answers (more strict about huge errors)
    # + centers the overall average performance around 0.5
    # + very simple formula
    # - is not exactly the perf displayed during the gamified experiment
    elif adjustment_type == EvalType.ADJUSTED:
        max_e = 0.55  # 0.5 in norm-1 is already a quite large error...
        max_t = 2.5 * allowed_time
        final_score = (1.0 - e / max_e) * (1.0 - t / max_t)
        return np.clip(final_score, 0.0, 1.0)

    # Bigger importance for the time perf. Gives the same overall average performance than the ADJUSTED.
    elif adjustment_type == EvalType.FOCUS_ON_TIME:
        max_e = 0.70
        max_t = 1.7 * allowed_time
        final_score = (1.0 - e / max_e) * (1.0 - t / max_t)
        return np.clip(final_score, 0.0, 1.0)

    # Bigger importance for the precisio perf. Gives the same overall average performance than the ADJUSTED.
    elif adjustment_type == EvalType.FOCUS_ON_ERROR:
        max_e = 0.5
        max_t = 4.0 * allowed_time
        final_score = (1.0 - e / max_e) * (1.0 - t / max_t)
        return np.clip(final_score, 0.0, 1.0)

    else:
        raise ValueError('The requested adjustement type does not exist.')


def expe4_ingame_eval(e, t, allowed_time):
    """ Score function used for real-time performance evaluation during the
    '4 parameters' experiment. Defined from early observations of alpha and beta experiments.
    e is the norm-1 parametric error, t is the total research duration.

    Note: the ADJUSTED performance must give very similar results. """

    # At first, we consider that the score is based on 2 independant performances: research time and final precision
    precision_term = 1.0 - 2.0*e  # null or negative if norm-1 error > 0.5
    time_term = 1.0 - t / (allowed_time*1.5)  # always positive, and is at least 1/3
    # Precision has a full 1.0 weight factor, while the time term has a 0.7 weight factor
    independant_score = precision_term + 0.70 * time_term  # might be negative if precision is very bad
    independant_score = np.clip(independant_score, 0.0, np.inf)  # (negative score are limited to zero)

    # Then, as the error is the most important, we'll multiply this temp result with a global precision factor
    # Without this: a random result at t=0 would always give a very good score (because of the time term)
    precision_factor = 1.0 - 1.3 * e  # null or negative if error >= 0.77 (which is a *huge* error)

    # Final score, with a 0.65 normalization factor
    final_score = 0.65 * independant_score * precision_factor

    final_score = np.clip(final_score, 0.0, 1.0)
    return final_score


class Analyzer:

    def __init__(self, expe):
        self.expe = expe
        self.allowed_time = expe.global_params.allowed_time

        self.mesh_resolution = 51
        self.max_displayed_e = 0.79  # e is the total error
        self.e_values = np.linspace(0.0, self.max_displayed_e, self.mesh_resolution)
        self.t_values = np.linspace(0.0, self.allowed_time, self.mesh_resolution)  # t is the research total duration
        self.e_grid, self.t_grid = np.meshgrid(self.e_values, self.t_values)
        self.elevation_angle = 25.0
        self.azimuth_angle = 45.0

        self.histogram_bins = np.linspace(0.0, 1.0, 25)
        self.kde_bw = 0.05  # Band Width of Gaussian Kernel Density Estimators

    def compare_adjusted(self, adj_types=[EvalType.ADJUSTED, EvalType.FOCUS_ON_TIME, EvalType.FOCUS_ON_ERROR]):

        fig = plt.figure(figsize=(9, 9))  # can't change the projection of an existing axes

        for i in range(len(adj_types)):
            ax_adj = fig.add_subplot(len(adj_types), 2, 1 + 2*i, projection='3d')
            # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
            surf = ax_adj.plot_surface(self.e_grid, self.t_grid, adjusted_eval(self.e_grid, self.t_grid,
                                                                               self.allowed_time,
                                                                               adjustment_type=adj_types[i]),
                                       linewidth=0, cmap=cm.plasma)
            self._configure_perf_surface_axes(ax_adj)
            name = adj_types[i].name.lower()
            if name:
                name = ' (' + name + ')'
            ax_adj.set(title='Perf. eval. function #{}{}'.format(str(adj_types[i].value), name),
                       xlabel='Norm-{} error $e$'.format(str(get_error_type_for_adjustment(adj_types[i]))))
            fig.colorbar(surf, aspect=18)

            ax_adj_hist = fig.add_subplot(len(adj_types), 2, 2 + 2*i)
            adjusted_s = self.expe.get_all_actual_s_1d(adjustment_type=adj_types[i])
            sns.distplot(adjusted_s, bins=self.histogram_bins,
                         kde=True, kde_kws={"bw": self.kde_bw}, ax=ax_adj_hist)
            ax_adj_hist.axvline(np.mean(adjusted_s), color='r', linestyle='--')
            plt.legend(['mean'])

            self._configure_perf_hist_kde_axes(ax_adj_hist, adjusted_s)

        plt.tight_layout()
        fig.subplots_adjust(left=0.05)
        figurefiles.save_in_figures_folder(fig, "Perf_adjusted_comparison_{}_{}_{}.pdf"
                                           .format(adj_types[0], adj_types[1], adj_types[2]))

    def plot_adjusted_perf_only(self):
        """ Evaluation function display for Nime20 paper """
        fig = plt.figure(figsize=(5, 2.5))
        ax_adj = fig.add_subplot(111, projection='3d')
        surf = ax_adj.plot_surface(self.e_grid, self.t_grid, adjusted_eval(self.e_grid, self.t_grid,
                                                                           self.allowed_time,
                                                                           adjustment_type=EvalType.ADJUSTED),
                                   linewidth=0, cmap=cm.plasma)
        self._configure_perf_surface_axes(ax_adj)  # many configs will be overriden just after
        ax_adj.set(xlabel='$e$, normalized sum of errors')
        ax_adj.set(ylabel='$d$, research duration [s]', zlabel=r'$s$, performance')
        ax_adj.set_xlim(0.0, 0.8)
        ax_adj.set_xticks(np.linspace(0.0, 0.8, 5))
        fig.colorbar(surf, aspect=18, pad=0.1)
        plt.tight_layout()
        fig.subplots_adjust(left=0.07, bottom=0.115, right=1.04)
        figurefiles.save_in_figures_folder(fig, "Perf_eval_adjusted.pdf")

    def _configure_perf_surface_axes(self, ax):
        ax.set_xlim(0.0, self.max_displayed_e)
        ax.set_ylim(0.0, self.allowed_time)
        ax.set_zlim(0.0, 1.0)
        ax.elev = self.elevation_angle
        ax.azim = self.azimuth_angle
        # y and z labels never change
        ax.set(ylabel='Research duration $d$ [s]', zlabel=r'Performance $s$')

    def _configure_perf_hist_kde_axes(self, ax, s_values):
        ax.set_xlim(0.0, 1.0)
        ax.set(title='Corresponding histogram and KDE',
               xlabel='Performances $s$ (sample of {} values)'.format(len(s_values)),
               ylabel='Normalized counts, estimated PDF')
