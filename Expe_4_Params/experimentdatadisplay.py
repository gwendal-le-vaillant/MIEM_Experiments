
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np

import experimentdataprocessing as edp


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
    plt.show()
    pass
