
# xet = Xml Element Tree. In the code : _et suffix means Element Tree
import xml.etree.ElementTree as xet
from distutils.util import strtobool
from enum import IntEnum

import numpy as np


class MethodType(IntEnum):
    NONE = -1
    FADER = 0   # independant control of independant parameters
    INTERP = 1  # interpolation between presets


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
        """ This constructors loads all data from files. This might take some
        time. """
        self.path_to_data = path_to_data
        print("--------------------------------")
        print("Loading experiments data from folder '{}'...".format(path_to_data))

        config_etree = xet.parse("{}A_OSC_Recorder_Experiment.xml".format(path_to_data))

        self.global_params = GlobalParameters(config_etree.getroot(), params_count, allowed_time)
        all_subjects_count = int(config_etree.getroot().find("count").text)

        synths_et = config_etree.getroot().find("synths")
        self.synths = list()
        for child in synths_et:
            self.synths.append(Synth(child, self.global_params))

        self.all_subjects = list()
        """ list of Subject:
        Represents all valid and unvalid (failed) tested subjects """
        for i in range(all_subjects_count):
            subject_info_et = xet.parse(self.get_subject_info_filename(i)).getroot()
            subject_data = np.genfromtxt(self.get_subject_data_filename(i), delimiter=";")
            self.all_subjects.append(Subject(subject_info_et, subject_data, self.global_params))
        self.subjects = [subject for subject in self.all_subjects if subject.is_valid]

        # display at the very end of loading (some data might be considered unvalid)
        print("...Data loading finished.")
        print(self)
        print("--------------------------------")

    def __str__(self):
        return "Experiment: {} valid subjects on {} tested. {}"\
            .format(len(self.subjects), len(self.all_subjects), self.global_params.__str__())

    def get_subject_data_filename(self, subject_index):
        return "{}Exp{:04d}_data.csv".format(self.path_to_data, subject_index)

    def get_subject_info_filename(self, subject_index):
        return "{}Exp{:04d}_info.xml".format(self.path_to_data, subject_index)


class GlobalParameters:
    """
    Inits the global parameters from the root child of the config element tree (from XML)
    """
    def __init__(self, etree_root, params_count, allowed_time):
        self.params_count = params_count
        self.allowed_time = allowed_time
        # synths
        synths_et = etree_root.find("synths")
        self.synths_count = int(synths_et.attrib["total_count"])
        self.synths_trial_count = int(synths_et.attrib["trials_count"])

        self.cycles_count = (2 * self.synths_non_trial_count) - self.synths_trial_count
        """ int: The number of cycles (called presets in the C++ code) that each subject should have performed
        within a full experiment.
        Contrainte : 1 essai pour chaque synthé de test, puis 2 essais pour chaque synthé réel
        """

    def __str__(self):
        return "Global parameters: {} synths ({} trial and {} non-trial) ; {} controllable parameters per synth"\
            .format(self.synths_count, self.synths_trial_count, self.synths_non_trial_count, self.params_count)

    @ property
    def synths_non_trial_count(self):
        return self.synths_count - self.synths_trial_count
    # translation of synths IDs, from experiment IDs (negative values) to processable positive indexes
    @ property
    def synth_id_offset(self):
        return 0 + self.synths_trial_count


class Synth:
    """
    Class that reads from an etree, and stores all general data related to a particular synth sound.
    """
    def __init__(self, etree, global_params):
        self.name = etree.attrib["name"]
        self.id = int(etree.attrib["id"])
        self.index = self.id + global_params.synth_id_offset
        # min/max values for all parameters
        self.min_value = float(etree.find("parameters").find("bounds").attrib["min"])
        self.max_value = float(etree.find("parameters").find("bounds").attrib["max"])
        # target value of each parameter
        self.target_values = np.zeros(global_params.params_count, dtype=np.double)
        for child in etree.find("parameters"):
            if child.tag == "bounds":
                continue
            param_id = int(child.attrib["id"])
            self.target_values[param_id-1] = float(child.attrib["target_value"])

    @ property
    def is_trial(self):
        return self.id < 0


class SexType(IntEnum):
    NON_BINARY = 0
    MALE = 1
    FEMALE = 2


class Subject:
    """
    Class for storing and processing all experiment data of a given subject

    On prévoit le stockage pour les experiences de tous les
    synthés, avec ou sans faders.
    Pour les presets de test la moitié de l'espace sera utilisé ; pour les autres,
    si l'expérience est complète, toutes les cases seront réellement utilisées
    """
    def __init__(self, info_et, raw_data, global_params):
        self.global_params = global_params # internal reference backup
        # - - - - - - - - Subject Info from XML file - - - - - - - -
        self.uid = int(info_et.attrib["UID"])
        self.is_valid = bool(strtobool(info_et.attrib["is_valid"]))
        first_questions_et = info_et.find("first_questions")
        final_questions_et = info_et.find("final_questions")
        self.remark = final_questions_et.find("remark").text
        # subject unvalid if some final data or data authorization is missing
        if (bool(strtobool(final_questions_et.attrib["answered"])) is not True) \
                or (bool(strtobool(first_questions_et.find("data_usage").attrib["allow"])) is not True):
            self.is_valid = False
        # also unvalid in case of hearing problems (usual vision troubles not considered critical)
        if bool(strtobool(first_questions_et.find("vision_impairment").attrib["checked"])):
            pass # print("[Subject {}]: vision impairment '{}'".format(self.uid, first_questions_et.find("vision_impairment").text))
        if bool(strtobool(first_questions_et.find("hearing_impairment").attrib["checked"])):
            print("[Subject {}]: hearing impairment '{}'".format(self.uid, first_questions_et.find("hearing_impairment").text))
            self.is_valid = False
        # ... end of data loaded in any case (valid or not valid)
        if self.is_valid is not True:
            return
        self.age = int(first_questions_et.find("age").text)
        sex_string = first_questions_et.find("sex").text.lower()
        if sex_string == "male":
            self.sex = SexType.MALE
        elif sex_string == "female":
            self.sex = SexType.FEMALE
        else:
            self.sex = SexType.NON_BINARY
        self.expertise_level = int(final_questions_et.find("expertise_level").text)
        self.methods_opinion = MethodsOpinion(final_questions_et.find("methods_opinion"))
        self.similar_interface = final_questions_et.find("similar_interface").text
        self.similar_experiment = final_questions_et.find("similar_expe").text
        # - - - - - - - - Data Info from XML file - - - - - - - -
        tested_presets_et = info_et.find("tested_presets")
        self.tested_cycles_count = int(tested_presets_et.attrib["count"])
        """ int: the number of cycles (presets) that were actually tested and recorded. Some might be unvalided later"""
        self.is_synth_valid = np.zeros((self.synths_count, 2), dtype=np.bool)
        self.synth_indexes_in_appearance_order = np.full((self.cycles_count, 1), -1000)
        self.synth_types_in_appearance_order   = np.full((self.cycles_count, 1), -1000)
        # We are going to access to cycles' child nodes by index, in order to throw errors if data is inconsistent
        for i in range(self.tested_cycles_count):
            is_valid = bool(strtobool(tested_presets_et[i].attrib["is_valid"]))
            if is_valid:
                # index in the main list of results
                synth_index = int(tested_presets_et[i].attrib["synth_id"]) + self.synth_id_offset
                # OK car renvoie 0 ou 1
                fader_or_interpolation = strtobool(tested_presets_et[i].attrib["from_interpolation"])
                self.is_synth_valid[synth_index, fader_or_interpolation] = True

        # - - - - - - - - Actual Data, from CSV file - - - - - - - -
        self.raw_data = raw_data

    @ property
    def synths_count(self):
        return self.global_params.synths_count

    @ property
    def cycles_count(self):
        return self.global_params.cycles_count

    @ property
    def synth_id_offset(self):
        return self.global_params.synth_id_offset


class MethodsOpinion:
        """
        Stores the final opinion on control methods from the XML node. See MethodType for the base enum type
        """
        def __init__(self, opinion_et):
            # MIGHT BE OPTIMIZED USING PANDAS
            self.fastest = self._convert_string_to_type(opinion_et.attrib["fastest"])
            self.most_precise = self._convert_string_to_type(opinion_et.attrib["most_precise"])
            self.most_intuitive = self._convert_string_to_type(opinion_et.attrib["most_intuitive"])
            self.preferred = self._convert_string_to_type(opinion_et.attrib["preferred"])
            pass

        @staticmethod
        def _convert_string_to_type(method_name):
            if method_name.lower() == "interpolation":
                return MethodType.INTERP
            elif method_name.lower() == "fader":
                return MethodType.FADER
            else:
                return MethodType.NONE

