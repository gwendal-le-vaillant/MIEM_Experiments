
import matplotlib.pyplot as plt
import numpy as np

import xml.etree.ElementTree as xet

import os

class Experiment:
    """
    Contains all results for all subjects of an experiment
     e.g. the "4 parameters experiments" can be fully stored inside this class

    4,5 dimensions pour l'expérience :
    1° i = Index du sujet d'expérience
    2° j = Numéro de synthé [de 0 à 11, y compris les essais, au lieu de -2 à 9]
       2,5° jbis = synthé via fader (indice 0) ou via interpolation (indice 1)
    3° k = Numéro de paramètre
    4° Time / recorded parameter values, variable height (2-column matrix)

    allowed_time is in seconds

    """
    def __init__(self, path_to_data="./", params_count=4, allowed_time=35.0):
        self.params_count = params_count
        self.allowed_time = allowed_time

        config_etree = xet.parse("{}A_OSC_Recorder_Experiment.xml".format(path_to_data))
        for root_child in config_etree.getroot():
            print(root_child.tag, root_child.attrib)




# test interne
if __name__ == "__main__":
    print("Miem expe data - auto run")
