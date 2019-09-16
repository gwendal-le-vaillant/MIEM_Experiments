

import gc  # forced garbage collection of old plots
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import experimentdataprocessing as edp
import figurefiles
import perfeval


def plot_age_and_sex(expe):
    female_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.FEMALE]
    male_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.MALE]
    other_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.NON_BINARY]

    fig, ax = plt.subplots(1, 1)
    plt.scatter(female_ages, np.full(len(female_ages), edp.SexType.FEMALE))
    plt.scatter(male_ages, np.full(len(male_ages), edp.SexType.MALE))
    plt.scatter(other_ages, np.full(len(other_ages), edp.SexType.NON_BINARY))
    ax.set(title="Age and sex of the {} subjects".format(len(expe.subjects)),
           xlabel="Age", ylabel="Sex")
    ax.set_ylim(-0.5, 2.5)
    ax.yaxis.set(ticks=range(0, 3), ticklabels=["non-binary", "male", "female"])
    plt.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()

    # save before show (because show empties the figure's internal data....)
    figurefiles.save_in_figures_folder(fig, "Age_and_sex.pdf")


class SubjectPerformancesVisualizer:
    """ Display all performances of a subject, with a slider for visualizing all results. """
    def __init__(self, expe, default_subject_index, show_radio_selector=True):

        self.expe = expe
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(7, 9))
        self.fig.subplots_adjust(bottom=0.15)

        # radio buttons for going through all subjects
        self.show_radio_selector = show_radio_selector
        radio_selector_x = 0.75 if show_radio_selector else 1.0
        self.fig.subplots_adjust(right=radio_selector_x-0.05)
        self.widget_ax = plt.axes([radio_selector_x, 0.1, 0.6, 0.8], frameon=True)  # ,aspect='equal')
        self.radio_buttons = RadioButtons(self.widget_ax, tuple([str(i) for i in range(len(self.expe.subjects))]))
        if show_radio_selector:
            self.fig.text(0.9, 0.92, 'Subject:', ha='center', fontsize='10')
        self.radio_buttons.on_clicked(self.on_radio_button_changed)
        self.radio_buttons.set_active(default_subject_index)

    def close(self):
        self.fig.clear()
        plt.close(self.fig)

    def on_radio_button_changed(self, label):
        subject_index = int(label)
        subject = self.expe.subjects[subject_index]
        self.update_plot(subject)

    def update_plot(self, subject):
        plt.suptitle("Durations $d_{ij}$, errors $e_{ij}$ and performances $s_{ij}$ for subject $i={}$"
                     + str(subject.index))

        synths_ids = self.expe.global_params.get_synths_ids()

        self.axes[0].clear()
        self.axes[0].set(ylabel="Research duration $d_{ij}$")
        self.axes[0].scatter(synths_ids, subject.d[:, 0], marker='s')
        self.axes[0].scatter(synths_ids, subject.d[:, 1], marker='D')
        self.axes[0].set_ylim([0, subject.global_params.allowed_time])  # hides the -1 unvalid values

        self.axes[1].clear()
        self.axes[1].set(ylabel="Norm-1 error $e_{ij}$")
        self.axes[1].scatter(synths_ids, subject.e_norm1[:, 0], marker='s')
        self.axes[1].scatter(synths_ids, subject.e_norm1[:, 1], marker='D')
        self.axes[1].set_ylim([0, 1])  # hides the -1 unvalid values
        self.axes[1].legend(['Faders', 'Interp'], loc="best")

        self.axes[2].clear()
        self.axes[2].set(ylabel="In-game displayed perf.")
        self.axes[2].scatter(synths_ids, subject.s_ingame[:, 0], marker='s')
        self.axes[2].scatter(synths_ids, subject.s_ingame[:, 1], marker='D')
        self.axes[2].set_ylim([0, 1])  # hides the -1 unvalid values
        self.axes[2].set_xlim([min(synths_ids)-0.5, max(synths_ids)+0.5])
        self.axes[2].xaxis.set_ticks(synths_ids)

        self.axes[3].clear()
        s_adj = subject.get_s_adjusted(adjustment_type=perfeval.EvalType.ADJUSTED)  # default best adjustment type
        self.axes[3].set(ylabel="Adjusted perf. $s_{ij}$", xlabel="Synth ID $j$")
        self.axes[3].scatter(synths_ids, s_adj[:, 0], marker='s')
        self.axes[3].scatter(synths_ids, s_adj[:, 1], marker='D')
        self.axes[3].set_ylim([0, 1])  # hides the -1 unvalid values
        self.axes[3].set_xlim([min(synths_ids)-0.5, max(synths_ids)+0.5])
        self.axes[3].xaxis.set_ticks(synths_ids)

        plt.draw()  # or does not update graphically...
        if not self.show_radio_selector:
            figurefiles.save_in_subjects_folder(self.fig, "Perf_subject_{:02d}.pdf".format(subject.index))


class SubjectCurvesVisualizer:
    """ Displays all curves (experiment data) recorded for a subject. Trial curves are included. """
    def __init__(self, expe, default_subject_index, show_radio_selector=True):
        assert expe.global_params.synths_count == 12, 'This display function is made for 12 synths (with 2 trials) at the moment'

        self.expe = expe

        self.rows_count = 4
        self.cols_count = 6
        self.fig, self.axes = plt.subplots(nrows=self.rows_count, ncols=self.cols_count, sharex=True, sharey=True,
                                           figsize=(14, 8))

        # global x and y labels, and limits
        self.fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize='12')
        self.fig.text(0.04, 0.5, 'Normalized parametric errors', va='center', rotation='vertical', fontsize='12')
        self.fig.subplots_adjust(bottom=0.1, left=0.1)

        # radio buttons for going through all subjects
        self.show_radio_selector = show_radio_selector
        radio_selector_x = 0.85 if show_radio_selector else 1.0
        self.fig.subplots_adjust(right=radio_selector_x - 0.05)
        self.widget_ax = plt.axes([radio_selector_x, 0.1, 0.3, 0.8], frameon=True)  # ,aspect='equal')
        self.radio_buttons = RadioButtons(self.widget_ax, tuple([str(i) for i in range(len(self.expe.subjects))]))
        if show_radio_selector:
            self.fig.text(0.93, 0.93, 'Subject:', ha='center', fontsize='10')
        self.radio_buttons.on_clicked(self.on_radio_button_changed)
        self.radio_buttons.set_active(default_subject_index)

    def close(self):
        self.fig.clear()
        plt.close(self.fig)

    def on_radio_button_changed(self, label):
        subject_index = int(label)
        subject = self.expe.subjects[subject_index]
        self.update_plot(subject)

    def update_plot(self, subject):
        synths = self.expe.synths

        self.fig.suptitle("Recorded data for subject $i={}$".format(subject.index))

        # actual plots of data
        for j in range(subject.tested_cycles_count):
            col = j % self.cols_count
            row = int( math.floor(j / self.cols_count) )
            self.axes[row, col].clear()

            synth_index = subject.synth_indexes_in_appearance_order[j]
            search_type = subject.search_types_in_appearance_order[j]
            if synth_index >= 0 and search_type >= 0 and subject.is_cycle_valid[synth_index, search_type]:
                if search_type == 0:
                    fader_or_interp_char = 'S'
                else:
                    fader_or_interp_char = 'I'
                self.axes[row, col].set_title('[{}]{}'.format(fader_or_interp_char, synths[synth_index].name), size=10)

                for k in range(subject.params_count):
                    self.axes[row, col].plot(subject.data[synth_index][search_type][k][:, 0],
                                             subject.data[synth_index][search_type][k][:, 1])
            else:
                self.axes[row, col].set_title('Unvalid data.', size=10)

        # remaining subplots are hidden
        for j in range(subject.tested_cycles_count, self.rows_count*self.cols_count):
            col = j % self.cols_count
            row = int( math.floor(j / self.cols_count) )
            self.axes[row, col].clear()
            self.axes[row, col].axis('off')

        # last is for drawing the legend
        for i in range(4):
            self.axes[self.rows_count - 1, self.cols_count - 1].plot([0], [0])
        self.axes[self.rows_count - 1, self.cols_count - 1].legend(['Parameter 1', 'Parameter 2', 'Parameter 3',
                                                                    'Parameter 4'], loc='lower right')

        # global x and y labels, and limits
        self.axes[0, 0].set_ylim([-1, 1])
        self.axes[0, 0].set_xlim([0, self.expe.global_params.allowed_time])

        # printing of the remark, and hearing impairments
        print('[Subject {:02d}] Remark=\'{}\''.format(subject.index, subject.remark))

        plt.draw()  # or does not update graphically...
        if not self.show_radio_selector:
            figurefiles.save_in_subjects_folder(self.fig, "Rec_data_subject_{:02d}.pdf".format(subject.index))


def save_all_subjects_to_pdf(expe):
    for subject in expe.subjects:
        curves_visualizer = SubjectCurvesVisualizer(expe, subject.index, show_radio_selector=False)
        perfs_visualizer = SubjectPerformancesVisualizer(expe, subject.index, show_radio_selector=False)
        curves_visualizer.close()
        perfs_visualizer.close()
        del curves_visualizer
        del perfs_visualizer
        gc.collect()


def all_perfs_violinplots(expe, perf_eval_type=perfeval.EvalType.ADJUSTED):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    #ax.violinplot(expe.get_all_actual_s_2d(perf_eval_type), showmedians=True, showmeans=True, showextrema=True)
    perfs_df = pd.DataFrame.from_records(expe.get_all_actual_s_2d(perf_eval_type))
    perfs_df.columns = ['Sliders', 'Interpolation']
    sns.violinplot(data=perfs_df, ax=ax, bw=0.15)
    ax.set_ylim(0.0, 1.0)
    ax.set(title="Performances of all subjects (eval. function = {})".format(perf_eval_type.name.lower()),
           xlabel="Research type", ylabel="Performance scores")
    plt.tight_layout()

    # TODO ANOVA, ou KS ?


def all_perfs_histogram(expe, perf_eval_type=perfeval.EvalType.ADJUSTED, display_KS=False):
    """ Shows all performances sorted in 2 groups (sliders and interp.), and displays
    the p-value of the Kolmogorov-Smirnov test """
    histogram_bins = np.linspace(0.0, 1.0, 20)
    kde_bw = 0.05

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    adjusted_s_2d = np.array(expe.get_all_actual_s_2d(perf_eval_type))
    distplot0 = sns.distplot(adjusted_s_2d[:, 0], bins=histogram_bins,
                             kde=True, kde_kws={"bw": kde_bw}, ax=ax, label='Sliders method')
    distplot1 = sns.distplot(adjusted_s_2d[:, 1], bins=histogram_bins,
                             kde=True, kde_kws={"bw": kde_bw}, ax=ax, label='Interp. method')
    ax.legend(loc='best')
    ax.set_xlim(0.0, 1.0)
    ax.set(title="Performances of all subjects (eval. function = {})".format(perf_eval_type.name.lower()),
           xlabel="Performance scores", ylabel="Normalized counts and estimated PDF")

    # Komolgorov-Smirnov test using scipy stats. The null hypothesis is 'the 2 samples are drawn from
    # the same distribution'. Null hypothesis can be rejected is p-value is small.
    # Obvious results.... p-value is around 10^-19
    if display_KS:
        [ks_stat, p_value] = stats.ks_2samp(adjusted_s_2d[:, 0], adjusted_s_2d[:, 1], alternative='two-sided')
        ax.text(x=0.1, y=0.1, s='KS-stat={}, p-value={}'.format(ks_stat, p_value),
                bbox=dict(boxstyle="round", fc="w"))

    plt.tight_layout()


def plot_all_perfs_per_synth(expe, plottype='box', perf_eval_type=perfeval.EvalType.ADJUSTED):

    assert expe.global_params.search_types_count == 2, 'This display allows slider/interp search types only'
    if plottype != 'box' and plottype != 'violin':
        raise ValueError('Only \'violin\' plot and \'box\' plot are available')

    all_s = expe.get_all_valid_s(perf_eval_type)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.set(title="Performances ({}) $s_{{ij}}$ of all subjects $i$, per synth $j$".format(perf_eval_type.name.lower()),
           xlabel="Synth ID $j$", ylabel="Performances $s_{ij}$")

    # box plot of all S perfs data, with empty space after each synth
    synths_range = range(expe.global_params.synths_trial_count*2, expe.global_params.synths_count * 2)
    cur_x_tick = 0
    x_ticks = []
    x_ticks_labels = []
    bps = []  # for box plots
    vls = []  # for violin plots
    for i in synths_range:
        if (cur_x_tick % 3 == 0):  # space
            ax.axvline(x=cur_x_tick, ymin=0.0, ymax=1.0, color='black', linewidth=0.5)
            x_ticks.append(cur_x_tick)
            x_ticks_labels.append(' ')
            cur_x_tick += 1

        # actual boxplot at every iteration
        synth_index = int(math.floor(float(i)/2.0))
        synth_id = synth_index-expe.global_params.synths_trial_count
        if (i%2) == 0:
            box_color = 'C0'
            x_ticks_labels.append('{} (sliders)'.format(synth_id))
        else:  # separating line after each synth
            box_color = 'C1'
            x_ticks_labels.append('{} (interp.)'.format(synth_id))

        if plottype == 'box':
            # artist costomization from https://matplotlib.org/3.1.0/gallery/statistics/boxplot.html
            median_props = dict(linestyle='-', linewidth=2.0, color='r')
            mean_point_props = dict(marker='D', markeredgecolor='black', markerfacecolor='r', markersize=4)
            bps.append(ax.boxplot(all_s[synth_index][i%2], positions=[cur_x_tick], sym='{}.'.format(box_color),
                                  widths=[0.6], showmeans=True, medianprops=median_props, meanprops=mean_point_props))
            plt.setp(bps[-1]['boxes'], color=box_color)
            plt.setp(bps[-1]['whiskers'], color=box_color)
            plt.setp(bps[-1]['fliers'], color=box_color)

        elif plottype == 'violin':
            vls.append(ax.violinplot(all_s[synth_index][i%2], positions=[cur_x_tick]))

        x_ticks.append(cur_x_tick)
        cur_x_tick += 1

    if plottype == 'box':
        ax.legend([bps[0]['boxes'][0], bps[1]['boxes'][0], bps[0]['medians'][0], bps[0]['means'][0]],
                  ['Sliders method', 'Interp. method', 'medians', 'means $\\overline{s_j}$'],
                  loc='center left', bbox_to_anchor=(1.0, 0.5))
    elif plottype == 'violin':
        pass  # not enough at the moment to really use a violin plot...

    ax.set_ylim([0, 1])
    ax.set_xlim([0, cur_x_tick])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, rotation=90)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    fig.tight_layout()
    figurefiles.save_in_figures_folder(fig, "Perfs_per_synth_{}-{}.pdf".format(plottype, perf_eval_type.value,
                                                                               perf_eval_type.name.lower()))


def plot_all_perfs_by_expertise(expe, perf_eval_type, add_swarm_plot=False):
    """ Builds box-plots of all performances, sorted by method and by expertise level """
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    s_df = expe.get_all_s_dataframe(perf_eval_type)
    sns.boxplot(x="expertise_level", y="performance", hue='search_type', data=s_df, ax=ax)
    if add_swarm_plot:
        sns.swarmplot(x="expertise_level", y="performance", hue='search_type', data=s_df, ax=ax,
                      dodge=True, size=3, color='black')

    ax.set(title="Performances sorted by expertise of subjects (eval. function = {})"
           .format(perf_eval_type.name.lower()))
    #fig.tight_layout()


def fit_perf_vs_expertise(expe, perf_eval_type, show_fit_analysis=False):
    assert len(expe.subjects[0].mean_s_ingame) == 2, 'Works for 2 methods only (fader + interp)'

    # Degrees of polynomial regressions
    faders_reg_degree = 2  # best is always 2 (in terms of R2 and RMSE)  # TODO vérifier si 1 ne suffit pas
    interp_reg_degree = 2  # TODO vérifier avec + de données si ordre 2 est nécessaire (1 peut être suffisant)

    expertise_levels = np.asarray([subject.expertise_level for subject in expe.subjects], dtype=int)
    # vstack of row arrays
    mean_s = np.vstack((np.asarray([subject.get_mean_s_adjusted(perf_eval_type)[0] for subject in expe.subjects]),
                        np.asarray([subject.get_mean_s_adjusted(perf_eval_type)[1] for subject in expe.subjects])))

    # manual polyfits, because seaborn does not (and will not...) give numerical outputs (only graphs, visualizations)
    if show_fit_analysis:
        reg0 = np.polyfit(expertise_levels, mean_s[0, :], faders_reg_degree)
        reg1 = np.polyfit(expertise_levels, mean_s[1, :], interp_reg_degree)
        reg_p = [np.poly1d(reg0), np.poly1d(reg1)]
        for i in range(2):
            plot_name = ('Sliders' if i == 0 else 'Interp')
            plot_name = plot_name + '_eval' + str(perf_eval_type.value)
            analyse_goodness_of_fit(expertise_levels, mean_s[i, :], reg_p[i], plot_name)

    # Seaborn fit graph (underlying functions: np.polyfit)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(title="Performances $\\overline{s_i}$ of subjects $i$, related to their expertise",
           xlabel="Estimated expertise level", ylabel="Mean performance $\\overline{s_i}$")

    regplot0 = sns.regplot(x=expertise_levels, y=mean_s[0, :], order=faders_reg_degree,
                           label="Sliders method", marker='s')
    regplot1 = sns.regplot(x=expertise_levels, y=mean_s[1, :], order=interp_reg_degree,
                           label="Interp method", marker='D')

    ax.set_ylim([0, 1])
    ax.set_xlim([min(expertise_levels)-0.5, max(expertise_levels)+0.5])

    ax.legend(loc='best')
    ax.text(x=0.8, y=0.1, s='Perf. evaluation type: {}'.format(perf_eval_type.name.lower()),
            bbox=dict(boxstyle="round", fc="w"))

    # fig.tight_layout()
    figurefiles.save_in_figures_folder(fig, "Perf_vs_expertise_eval{}.pdf".format(perf_eval_type))


def analyse_goodness_of_fit(x_data, y_data, poly_fit, fit_name):
    """
    Computation of SSE, R-square, adjusted R-square and RMSE, and display of residuals histogram
    (which should be close to a normal distribution)
    gof statistics: https://fr.mathworks.com/help/curvefit/evaluating-goodness-of-fit.html
    """

    # useful display and computational data
    y_fitted = poly_fit(x_data)
    min_x_display = min(x_data) - abs(max(x_data) - min(x_data)) * 0.1
    max_x_display = max(x_data) + abs(max(x_data) - min(x_data)) * 0.1
    x_fitted_display = np.linspace(min_x_display, max_x_display)
    y_fitted_display = poly_fit(x_fitted_display)

    # goodness of fit indicators
    dof = len(y_data) - (poly_fit.order+1)  # degrees of freedom
    SSE = np.sum((y_data - y_fitted) ** 2)  # Sum of Squared Errors
    SST = np.sum((y_data - np.mean(y_data)) ** 2)  # Total Sum of Squares (about the mean)
    R2 = 1.0 - SSE/SST  # R squared
    RMSE = math.sqrt( SSE / dof )  # Root Mean Squared Error

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)

    # plot of the fitted polynomial itself
    ax.plot(x_fitted_display, y_fitted_display, color='C6')
    ax.scatter(x_data, y_data, color='C7')
    ax.set(title="Fitted polynomial (deg{}) for \'{}\'".format(poly_fit.order, fit_name), xlabel="x", ylabel="y")

    # histogram of residuals
    ax2 = fig.add_subplot(1, 2, 2)
    sns.distplot(y_data - y_fitted, kde=True, ax=ax2)
    ax2.set(title="Histogram of residuals", xlabel="Residual value $y_k - \\widehat{y_k}$", ylabel="Count")

    # display of fit indicators
    fig.text(0.02, 0.02, '$SSE = {0:.6f}$'.format(SSE), fontsize='10')
    fig.text(0.27, 0.02, '$R^2 = {0:.6f}$'.format(R2), fontsize='10')
    fig.text(0.52, 0.02, '$RMSE = {0:.6f}$'.format(RMSE), fontsize='10')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    figurefiles.save_in_perfs_fits_folder(fig, "Polyfit_{}_order_{}.pdf".format(fit_name, poly_fit.order))


def plot_opinions_on_methods(expe):

    # We rely on a pre-computed pandas dataframe for this
    ax = expe.opinions.plot.bar(rot=0)
    ax.set(title='Answers to the questions: which method was the [...] ?',
           ylabel='Amount of subjects', xlabel='Characteristic asked')
    # legend needs more space
    max_displayed_y = int(math.floor( expe.opinions.max().max() * 1.4 ))

    ax.legend(loc='best')
    ax.set_ylim([0, max_displayed_y])

    figurefiles.save_in_figures_folder(plt.gcf(), "Opinions_on_methods.pdf")
