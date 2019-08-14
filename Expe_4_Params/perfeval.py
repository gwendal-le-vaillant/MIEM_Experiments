
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Color Maps

import experimentdataprocessing as edp
import figurefiles


def get_best_type():
    return 3


def get_error_type_for_adjustement(adjustement_type):
    """ Returns the norm (1 or 2) to be used with the corresponding adjusted performance evaluation function.

    Norm-1 is generally preferred because controlled parameters produced very independant results ; it seems
     more reasonable to simply add all the errors and normalize the result,
     rather than taking the root mean square of all errors.
     """
    if adjustement_type == 0:
        return 1

    elif adjustement_type == 1:
        return 1

    elif adjustement_type == 2:
        return 2

    elif adjustement_type == 3:
        return 1

    else:
        raise ValueError('The requested adjustement type does not exist.')


def adjusted_eval(e, t, allowed_time, adjustement_type=get_best_type()):
    """ Computes adjusted performance values ; the type of adjustement (int) can be chosen, including the
    norm to be used for error computation. """
    if adjustement_type == 0:
        return expe4_ingame_eval(e, t, allowed_time)

    elif adjustement_type == 1:
        precision_term = 1.0 - 2.0 * e  # null or negative if total error > 0.5
        time_term = 1.0 - t / allowed_time  # always positive, and is at least 1/3
        final_score = 0.6 * precision_term + 0.4 * time_term
        return np.clip(final_score, 0.0, 1.0)

    elif adjustement_type == 2:
        max_allowed_e = 0.55  # 0.5 in norm-1 is already a huge error (corresponding to a very bad score)
        precision_term = 1.0 - e / max_allowed_e
        time_term = 1.0 - t / (5.0*allowed_time)  # always positive, and is at least 80%
        final_score = precision_term * time_term
        return np.clip(final_score, 0.0, 1.0)

    elif adjustement_type == 3:
        return adjusted_eval(e, t, allowed_time, adjustement_type=2)

    else:
        raise ValueError('The requested adjustement type does not exist.')


def expe4_ingame_eval(e, t, allowed_time):
    """ Score function used for real-time performance evaluation during the
    '4 parameters' MIEM experiment. Defined from early observations of alpha and beta experiments.
    e is the norm-1 parametric error, t is the total research duration. """

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
        self.elevation_angle = 20.0
        self.azimuth_angle = 30.0

        self.histogram_bins = np.linspace(0.0, 1.0, 25)
        self.kde_bw = 0.05  # Band Width of Gaussian Kernel Density Estimators

    def compare_ingame_to_adjusted(self):
        s_ingame_z_values = expe4_ingame_eval(self.e_grid, self.t_grid, self.allowed_time)
        # best type will be chosen by default
        s_adjusted_z_values = adjusted_eval(self.e_grid, self.t_grid, self.allowed_time)

        fig = plt.figure(figsize=(9, 6))  # can't change the projection of an existing axes

        # - - - IN GAME - - -
        # Performance function visualized as a surface
        ax_ingame = fig.add_subplot(2, 2, 1, projection='3d')
        # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
        surf = ax_ingame.plot_surface(self.e_grid, self.t_grid, s_ingame_z_values,
                                 # rstride=1, cstride=1,
                                 linewidth=0, cmap=cm.plasma)
        self._configure_perf_surface_axes(ax_ingame)
        ax_ingame.set(title='In-game perf. evaluation function', xlabel='Norm-1 error $e$')
        fig.colorbar(surf, aspect=18)

        # Histogram + gaussian kernel density estimate of result performances
        ax_ingame_hist = fig.add_subplot(2, 2, 2)
        ingame_s = self.expe.all_actual_ingame_s_1d
        sns.distplot(ingame_s, bins=self.histogram_bins,
                     kde=True, kde_kws={"bw": self.kde_bw}, ax=ax_ingame_hist)
        self._configure_perf_hist_kde_axes(ax_ingame_hist, ingame_s)

        # - - - ADJUSTED - - -
        ax_adj = fig.add_subplot(2, 2, 3, projection='3d')
        # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
        surf = ax_adj.plot_surface(self.e_grid, self.t_grid, s_adjusted_z_values,
                                 # rstride=1, cstride=1,
                                 linewidth=0, cmap=cm.plasma)
        self._configure_perf_surface_axes(ax_adj)
        ax_adj.set(title='Adjusted perf. evaluation function',
                   xlabel='Norm-{} error $e$'.format(str(get_error_type_for_adjustement(get_best_type()))))
        fig.colorbar(surf, aspect=18)

        ax_adj_hist = fig.add_subplot(2, 2, 4)
        adjusted_s = self.expe.get_all_actual_adjusted_s_1d()  # best type by default
        sns.distplot(adjusted_s, bins=self.histogram_bins,
                     kde=True, kde_kws={"bw": self.kde_bw}, ax=ax_adj_hist)
        self._configure_perf_hist_kde_axes(ax_adj_hist, adjusted_s)

        plt.tight_layout()
        fig.subplots_adjust(left=0.05)
        figurefiles.save_in_figures_folder(fig, "Perf_adjusted_vs_ingame.pdf")

    def compare_adjusted(self):
        adj_types = [0, 2, 3]

        fig = plt.figure(figsize=(9, 9))  # can't change the projection of an existing axes

        for i in range(len(adj_types)):
            ax_adj = fig.add_subplot(len(adj_types), 2, 1 + 2*i, projection='3d')
            # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
            surf = ax_adj.plot_surface(self.e_grid, self.t_grid, adjusted_eval(self.e_grid, self.t_grid, self.allowed_time,
                                                                               adjustement_type=adj_types[i]),
                                       linewidth=0, cmap=cm.plasma)
            self._configure_perf_surface_axes(ax_adj)
            ax_adj.set(title='Perf. evaluation function #{}'.format(str(adj_types[i])),
                       xlabel='Norm-{} error $e$'.format(str(get_error_type_for_adjustement(adj_types[i]))))
            fig.colorbar(surf, aspect=18)

            ax_adj_hist = fig.add_subplot(len(adj_types), 2, 2 + 2*i)
            adjusted_s = self.expe.get_all_actual_adjusted_s_1d(adjustement_type=adj_types[i])
            sns.distplot(adjusted_s, bins=self.histogram_bins,
                         kde=True, kde_kws={"bw": self.kde_bw}, ax=ax_adj_hist)
            self._configure_perf_hist_kde_axes(ax_adj_hist, adjusted_s)

        plt.tight_layout()
        fig.subplots_adjust(left=0.05)
        figurefiles.save_in_figures_folder(fig, "Perf_adjusted_comparison.pdf")

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
               ylabel='Normalized counts and estimated PDF')
