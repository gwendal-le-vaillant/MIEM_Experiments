
import os


figures_folder = "./Figures"
subjects_subfolder = "Subjects"
perfs_fits_subfolder = "Perfs_fits"


def try_create_figures_folder():
    if not os.path.exists(figures_folder):
        os.mkdir(figures_folder)


def save_in_figures_folder(fig, filename):
    try_create_figures_folder()
    fig.savefig("{}/{}".format(figures_folder, filename))


def save_in_subjects_folder(fig, filename):
    try_create_figures_folder()
    subjects_folder = "{}/{}".format(figures_folder, subjects_subfolder)
    if not os.path.exists(subjects_folder):
        os.mkdir(subjects_folder)
    fig.savefig("{}/{}".format(subjects_folder, filename))


def save_in_perfs_fits_folder(fig, filename):
    try_create_figures_folder()
    perfs_folder = "{}/{}".format(figures_folder, perfs_fits_subfolder)
    if not os.path.exists(perfs_folder):
        os.mkdir(perfs_folder)
    fig.savefig("{}/{}".format(perfs_folder, filename))