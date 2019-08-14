
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Color Maps

import experimentdataprocessing as edp
import figurefiles


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
    precision_factor = 1.0 - 1.3 * e  # null or negative if error >= 0.77 (which is a *huge* error)

    # Final score, with a 0.65 normalization factor
    final_score = 0.65 * independant_score * precision_factor

    final_score = np.clip(final_score, 0.0, 1.0)
    return final_score


def adjusted_eval(e, t, allowed_time):  ## TODO changer pour de vrai....
    # At first, we consider that the score is based on 2 independant performances: research time and final precision
    precision_term = 1.0 - 2.0*e  # null or negative if norm-1 error > 0.5
    time_term = 1.0 - t / (allowed_time*1.5)  # always positive, and is at least 1/3
    # Precision has a full 1.0 weight factor, while the time term has a 0.7 weight factor
    independant_score = precision_term + 0.70 * time_term  # might be negative if precision is very bad
    independant_score = np.clip(independant_score, 0.0, np.inf)  # (negative score are limited to zero)
    # Then, as the error is the most important, we'll multiply this temp result with a global precision factor
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

    def compare_ingame_to_adjusted(self):
        s_ingame_values = expe4_ingame_eval(self.e_grid, self.t_grid, self.allowed_time)

        # Performance function visualized as a surface
        fig = plt.figure(figsize=(12, 8))  # can't change the projection of an existing axes
        ax00 = fig.add_subplot(2, 2, 1, projection='3d')
        # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
        surf = ax00.plot_surface(self.e_grid, self.t_grid, s_ingame_values,
                                 # rstride=1, cstride=1,
                                 linewidth=0, cmap=cm.plasma)
        self._configure_perf_surface_axes(ax00)
        ax00.set(title='In-game performance evaluation function', xlabel=r'Normalized norm-1 error $e$')
        fig.colorbar(surf, aspect=18)

        # Histogram + gaussian kernel density estimate of result performances
        ax01 = fig.add_subplot(2, 2, 2)
        # quite small bandwidth for the kde
        sns.distplot(self.expe.all_actual_ingame_s_1d, bins=self.histogram_bins,
                     kde=True, kde_kws={"bw": 0.05}, ax=ax01)
        self._configure_perf_hist_kde_axes(ax01)

        plt.tight_layout()

        figurefiles.save_in_figures_folder(fig, "Perf_eval_function.pdf")

    def _configure_perf_surface_axes(self, ax):
        ax.set_xlim(0.0, self.max_displayed_e)
        ax.set_ylim(0.0, self.allowed_time)
        ax.set_zlim(0.0, 1.0)
        ax.elev = self.elevation_angle
        ax.azim = self.azimuth_angle
        # y and z labels never change
        ax.set(ylabel='Research duration $d$ [s]', zlabel=r'Performance $s$')

    def _configure_perf_hist_kde_axes(self, ax):
        ax.set_xlim(0.0, 1.0)
        ax.set(title='Corresponding histogram and kernel density estimate',
               xlabel=r'Performances $s$', ylabel='Normalized counts and estimated PDF')
