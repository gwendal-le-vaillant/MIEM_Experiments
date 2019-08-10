
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Color Maps

import experimentdataprocessing as edp

figures_save_folder = "./Figures"


def display_age_and_sex(expe):
    female_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.FEMALE]
    male_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.MALE]
    other_ages = [subject.age for subject in expe.subjects if subject.sex == edp.SexType.NON_BINARY]

    fig, ax = plt.subplots(1, 1)
    plt.scatter(female_ages, np.full(len(female_ages), edp.SexType.FEMALE))
    plt.scatter(male_ages, np.full(len(male_ages), edp.SexType.MALE))
    plt.scatter(other_ages, np.full(len(other_ages), edp.SexType.NON_BINARY))
    ax.set(title="Age and sex of subjects", xlabel="Age", ylabel="Sex")
    ax.set_ylim(-0.5, 2.5)
    ax.yaxis.set(ticks=range(0, 3), ticklabels=["non-binary", "male", "female"])
    plt.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()

    # save before show (because show seems to empty the figure)
    plt.savefig("{}/Age_and_sex.pdf".format(figures_save_folder))


def display_performance_score_surface(params_count=4, allowed_time=35.0):

    mesh_resolution = 51
    max_displayed_e = 0.79  # e is the total error
    e_values = np.linspace(0.0, max_displayed_e, mesh_resolution)
    t_values = np.linspace(0.0, allowed_time, mesh_resolution)  # t is the research total duration
    e_grid, t_grid = np.meshgrid(e_values, t_values)

    # CODE EXEMPLE : source = https://hub.packtpub.com/creating-2d-3d-plots-using-matplotlib/
    p_values = edp.score_expe4(e_grid, t_grid, allowed_time)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(e_grid, t_grid, p_values,
                           #rstride=1, cstride=1,
                           linewidth=0, cmap=cm.plasma)
    ax.set_xlim(0.0, max_displayed_e)
    ax.set_ylim(0.0, allowed_time)
    ax.set_zlim(0.0, 1.0)
    ax.elev = 20.0
    ax.azim = 30.0
    ax.set(title='In-game performance evaluation function', xlabel=r'Normalized total error $E$', ylabel='Research duration $T$ [s]', zlabel=r'Performance $P$')
    fig.colorbar(surf, aspect=18)
    plt.tight_layout()

    plt.savefig("{}/Perf_eval_function.pdf".format(figures_save_folder))


class SubjectPerformancesVisualizer:
    """ Display all performances of a subject, with a slider for visualizing all results. """
    def __init__(self, expe, default_subject_index, show_radio_selector=True):

        self.expe = expe
        self.fig, self.axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, figsize=(7, 6))
        self.fig.subplots_adjust(bottom=0.15)

        # radio buttons for going through all subjects
        radio_selector_x = 0.75 if show_radio_selector else 1.5
        self.fig.subplots_adjust(right=radio_selector_x-0.05)
        self.widget_ax = plt.axes([radio_selector_x, 0.1, 0.6, 0.8], frameon=True)  # ,aspect='equal')
        self.radio_buttons = RadioButtons(self.widget_ax, tuple([str(i) for i in range(len(self.expe.subjects))]))
        if show_radio_selector:
            self.fig.text(0.9, 0.92, 'Subject:', ha='center', fontsize='10')
        self.radio_buttons.on_clicked(self.on_radio_button_changed)
        self.radio_buttons.set_active(default_subject_index)

    def on_radio_button_changed(self, label):
        subject_index = int(label)
        subject = self.expe.subjects[subject_index]
        self.update_plot(subject)

    def update_plot(self, subject):
        plt.suptitle("Durations T, errors E and performances P for subject $i={}$".format(subject.index))

        synths_count = self.expe.global_params.synths_count
        synths_indexes = np.arange(synths_count)

        self.axes[0].clear()
        self.axes[0].set(ylabel="Error $E$")
        self.axes[0].scatter(synths_indexes, subject.E[:, 0], marker='s')
        self.axes[0].scatter(np.arange(subject.global_params.synths_count), subject.E[:, 1], marker='D')
        self.axes[0].set_ylim([0, 1])  # hides the -1 unvalid values

        self.axes[1].clear()
        self.axes[1].set(ylabel="Research duration $T$")
        self.axes[1].scatter(synths_indexes, subject.T[:, 0], marker='s')
        self.axes[1].scatter(np.arange(subject.global_params.synths_count), subject.T[:, 1], marker='D')
        self.axes[1].set_ylim([0, subject.global_params.allowed_time])  # hides the -1 unvalid values

        self.axes[2].clear()
        self.axes[2].set(ylabel="Performance $P$", xlabel="Synth ID")
        self.axes[2].scatter(synths_indexes, subject.P[:, 0], marker='s')
        self.axes[2].scatter(np.arange(subject.global_params.synths_count), subject.P[:, 1], marker='D')
        self.axes[2].set_ylim([0, 1])  # hides the -1 unvalid values
        self.axes[2].set_xlim([-0.5, synths_count-0.5])
        self.axes[2].xaxis.set_ticks(synths_indexes)

        plt.savefig("{}/Perf_subject_{:02d}.pdf".format(figures_save_folder, subject.index))


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
        radio_selector_x = 0.9 if show_radio_selector else 1.5
        self.fig.subplots_adjust(right=radio_selector_x - 0.05)
        self.widget_ax = plt.axes([radio_selector_x, 0.1, 0.3, 0.8], frameon=True)  # ,aspect='equal')
        self.radio_buttons = RadioButtons(self.widget_ax, tuple([str(i) for i in range(len(self.expe.subjects))]))
        if show_radio_selector:
            self.fig.text(0.93, 0.93, 'Subject:', ha='center', fontsize='10')
        self.radio_buttons.on_clicked(self.on_radio_button_changed)
        self.radio_buttons.set_active(default_subject_index)

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
                    fader_or_interp_char = 'F'
                else:
                    fader_or_interp_char = 'I'
                self.axes[row, col].set_title('{} {}'.format(fader_or_interp_char, synths[synth_index].name), size=10)

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

        plt.draw()  # or does not update graphically...

        self.fig.savefig("{}/Rec_data_subject_{:02d}.pdf".format(figures_save_folder, subject.index))
