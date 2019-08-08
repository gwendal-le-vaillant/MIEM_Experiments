import matplotlib.pyplot as plt
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
    plt.show()


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

    plt.savefig("{}/Performance evaluation function.pdf".format(figures_save_folder))
    plt.show()

    pass

