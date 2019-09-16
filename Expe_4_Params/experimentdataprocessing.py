
import time
import pickle
# xet = Xml Element Tree. In the code : _et suffix means Element Tree
import xml.etree.ElementTree as xet
from distutils.util import strtobool
from enum import IntEnum

import statistics
import numpy as np
import pandas as pd


import perfeval


class MethodType(IntEnum):
    NONE = -1
    SLIDERS = 0   # independant control of independant parameters
    INTERP = 1  # interpolation between presets


def load_experiment_once_a_day(data_path, force_reload=False):
    """
    Optimized loading (XML/CSV files will be parsed and pickled only once a day)

    :param data_path: the path to XML and CSV files of this experiment
    :param force_reload: indicates whether the experiment must be reloaded, even if recent pickeled data is available
    :return: the Experiment class instance
    """
    pickle_filename = "{}/python_experiment_class.pickle".format(data_path)
    if not force_reload:
        try:
            pickle_file = open(pickle_filename, mode='rb')
            print("Loading data from previous pickled file...")
            experiment_instance = pickle.load(pickle_file, encoding='latin-1')
            is_pickle_file_outdated = experiment_instance.construction_time < (time.time() - (3600 * 24) )
            if is_pickle_file_outdated:
                print("Pickled data is outdated.")
            else:
                delta_time = time.time() - experiment_instance.construction_time
                print("Ok, pickle data has been constructed {:2.1f} hours ago.".format(delta_time / 3600.0))
        except FileNotFoundError:
            print("No pickle file found -> loading fresh data from CSV and XML files")
            is_pickle_file_outdated = True
        except Exception:
            print("Unknown exception: pickle data file seems corrupted. Reloading experiment.")
            is_pickle_file_outdated = True

    if force_reload or is_pickle_file_outdated:
        experiment_instance = Experiment(data_path)
        pickle_file = open(pickle_filename, mode='wb')
        pickle.dump(experiment_instance, pickle_file, protocol=pickle.DEFAULT_PROTOCOL)

    return experiment_instance


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
        self.construction_time = time.time()
        print("--------------------------------")
        print("Loading experiments data from folder '{}'...".format(path_to_data))

        config_etree = xet.parse("{}A_OSC_Recorder_Experiment.xml".format(path_to_data))

        self.global_params = GlobalParameters(config_etree.getroot(), params_count, allowed_time)
        all_subjects_count = int(config_etree.getroot().find("count").text)

        # Getting all synth info
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
            self.all_subjects.append(Subject(subject_info_et, subject_data, self.global_params, self.synths))
        self.subjects = [subject for subject in self.all_subjects if subject.is_valid]
        for i in range(len(self.subjects)):
            self.subjects[i].set_index(i)

        # Construction of a dataframe for opinions
        self.opinions_per_synth = pd.DataFrame({'fastest': [subject.methods_opinion.fastest
                                                              for subject in self.subjects],
                                                  'most precise': [subject.methods_opinion.most_precise
                                                                   for subject in self.subjects],
                                                  'most intuitive': [subject.methods_opinion.most_intuitive
                                                                     for subject in self.subjects],
                                                  'preferred': [subject.methods_opinion.preferred
                                                                for subject in self.subjects]})
        # dataframe to be displayed directly in a bar plot - index will be the absissa
        faders_counts = [len([0 for subject in self.subjects if subject.methods_opinion.fastest == MethodType.SLIDERS]),
                         len([0 for subject in self.subjects if subject.methods_opinion.most_precise == MethodType.SLIDERS]),
                         len([0 for subject in self.subjects if subject.methods_opinion.most_intuitive == MethodType.SLIDERS]),
                         len([0 for subject in self.subjects if subject.methods_opinion.preferred == MethodType.SLIDERS])]
        interp_counts = [len([0 for subject in self.subjects if subject.methods_opinion.fastest == MethodType.INTERP]),
                         len([0 for subject in self.subjects if subject.methods_opinion.most_precise == MethodType.INTERP]),
                         len([0 for subject in self.subjects if subject.methods_opinion.most_intuitive == MethodType.INTERP]),
                         len([0 for subject in self.subjects if subject.methods_opinion.preferred == MethodType.INTERP])]
        none_counts = [len([0 for subject in self.subjects if subject.methods_opinion.fastest == MethodType.NONE]),
                       len([0 for subject in self.subjects if subject.methods_opinion.most_precise == MethodType.NONE]),
                       len([0 for subject in self.subjects if subject.methods_opinion.most_intuitive == MethodType.NONE]),
                       len([0 for subject in self.subjects if subject.methods_opinion.preferred == MethodType.NONE])]
        self.opinions = pd.DataFrame({'Faders': faders_counts,
                                      'Interpolation': interp_counts,
                                      'None': none_counts},
                                     index=['fastest', 'most precise', 'most intuitive', 'preferred'])

        # display at the very end of loading (some data might be considered unvalid)
        print("...Data loading finished.")
        print("--------------------------------")

    def __str__(self):
        return "Experiment: {} valid subjects on {} tested. {}"\
            .format(len(self.subjects), len(self.all_subjects), self.global_params.__str__())

    def get_subject_data_filename(self, subject_index):
        return "{}Exp{:04d}_data.csv".format(self.path_to_data, subject_index)

    def get_subject_info_filename(self, subject_index):
        return "{}Exp{:04d}_info.xml".format(self.path_to_data, subject_index)

    def precompute_adjusted_s(self):
        """ Precomputes adjusted performances of all subjects, and stores
         all results in pandas dataframes (one df for each perf_eval_type) """
        for subject in self.subjects:
            subject.precompute_adjusted_s()
        # Construction of the main pandas dataframes
        self.all_s_dataframes = list()
        for eval_type in perfeval.EvalType:
            list_of_dataframes = list()  # list of all subjects' dataframes (will be concatenated)
            if eval_type != perfeval.EvalType.COUNT:
                # construction of lists of lists of perfs, subject id, etc....
                for subject in self.subjects:
                    list_of_dataframes.append( subject.get_all_s_dataframe(eval_type) )
                self.all_s_dataframes.append(pd.concat(list_of_dataframes))  # 1 dataframe for 1 perf eval type

    def get_all_valid_s(self, perf_eval_type=perfeval.EvalType.ADJUSTED):
        """ Returns a 3D list containing perf results of all subjects, indexed by synth idx and search type """
        all_s = [ [ [] for j2 in range(self.global_params.search_types_count)]
                       for j in range(self.global_params.synths_count) ]
        for subject in self.subjects:
            subject_s_adj = subject.get_s_adjusted(perf_eval_type)
            for j in range(self.global_params.synths_count):
                for j2 in range(self.global_params.search_types_count):
                    if subject_s_adj[j, j2] >= 0.0:  # only valid recorded performances are considered
                        all_s[j][j2] += [subject_s_adj[j, j2]]
        return all_s

    def get_all_valid_s(self, perf_eval_type=perfeval.EvalType.ADJUSTED):
        """ Returns a 3D list containing perf results of all subjects, indexed by synth idx and search type """
        all_s = [ [ [] for j2 in range(self.global_params.search_types_count)]
                       for j in range(self.global_params.synths_count) ]
        for subject in self.subjects:
            subject_s_adj = subject.get_s_adjusted(perf_eval_type)
            for j in range(self.global_params.synths_count):
                for j2 in range(self.global_params.search_types_count):
                    if subject_s_adj[j, j2] >= 0.0:  # only valid recorded performances are considered
                        all_s[j][j2] += [subject_s_adj[j, j2]]
        return all_s

    def get_all_actual_s_2d(self, perf_eval_type=perfeval.EvalType.ADJUSTED):
        """ Returns a 2D list containing perf results of all subjects, indexed by search type (interp/fader) only """
        all_s = list()
        for subject in self.subjects:
            subject_s_adj = subject.get_s_adjusted(perf_eval_type)
            for j in range(self.global_params.synths_count):
                temp_perfs = []
                for j2 in range(self.global_params.search_types_count):
                    temp_perfs += [subject_s_adj[j, j2]]
                if min(temp_perfs) >= 0.0:  # only full-valid recorded performances are considered
                    all_s.append( temp_perfs )
        return all_s

    def get_all_actual_s_1d(self, adjustment_type):
        """ Returns a flattend 1D numpy array of all perfs. Does not include trial or unvalid perfs. """
        flattened_s = []  # because it involves memory allocations: use of a python list...
        for subject in self.subjects:
            flattened_s.extend(subject.get_actual_s_adjusted_1d(adjustment_type))
        return np.array(flattened_s)

    def get_all_s_dataframe(self, perf_eval_type=perfeval.EvalType.ADJUSTED):
        return self.all_s_dataframes[int(perf_eval_type)]


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

        self.search_types_count = 2
        """ Currently, 2 values available: fader (0) or interpolation (1) """
        self.cycles_count = (self.search_types_count * self.synths_non_trial_count) + self.synths_trial_count
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

    def get_synths_ids(self):
        return range(-self.synths_trial_count, self.synths_non_trial_count)


class Synth:
    """
    Class that reads from an etree, and stores all general data related to a particular synth sound.
    """
    def __init__(self, etree, global_params):
        self.name = etree.attrib["name"]
        self.id = int(etree.attrib["id"])
        self.index = self.id + global_params.synth_id_offset
        # min/max values : for now, they are the for all parameters
        self.min_values = np.full((global_params.params_count, 1),
                                  float(etree.find("parameters").find("bounds").attrib["min"]))
        self.max_values = np.full((global_params.params_count, 1),
                                  float(etree.find("parameters").find("bounds").attrib["max"]))
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
    def __init__(self, info_et, raw_data, global_params, synths):
        self.global_params = global_params # internal reference backup
        self.index = -1 # to be defined after construction of all subjects
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
            pass  # print("[Subject {}]: vision impairment '{}'".format(self.uid, first_questions_et.find("vision_impairment").text))
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
        """ int: The number of cycles (presets) that were actually tested and recorded. Some might be unvalided later"""
        self.tested_cycles_count = int(tested_presets_et.attrib["count"])
        """ 2d-array indicating wether a cycle (accessible via synth index / search type) is valid or not"""
        self.is_cycle_valid = np.zeros((self.synths_count, 2), dtype=np.bool)
        """ The n-th element of this list gives the index of the n-th synth research """
        self.synth_indexes_in_appearance_order = [-1000] * self.cycles_count
        """ The n-th element of this list gives the type of the n-th synth research (fader or interpolation) """
        self.search_types_in_appearance_order  = [-1000] * self.cycles_count
        # We are going to access to cycles' child nodes by index, in order to throw errors if data is inconsistent
        for i in range(self.tested_cycles_count):
            is_valid = bool(strtobool(tested_presets_et[i].attrib["is_valid"]))
            if is_valid:
                # index in the main list of results
                synth_index = int(tested_presets_et[i].attrib["synth_id"]) + self.synth_id_offset
                # OK car renvoie 0 ou 1
                fader_or_interpolation = strtobool(tested_presets_et[i].attrib["from_interpolation"])
                self.is_cycle_valid[synth_index, fader_or_interpolation] = True
                # backup of synth index and type, in appearance order
                self.synth_indexes_in_appearance_order[i] = synth_index
                self.search_types_in_appearance_order[i] = fader_or_interpolation

        # - - - - - - - - Actual Data, from CSV file - - - - - - - -
        # pre-allocation of data lists - will be a 2,5 D list itself (synth, interp, paramId),
        # but each actual element of the list will be a numpy matrix (recorded time and values)
        self.data = [ [ [ [] for k in range(0, self.params_count) ] for jbis in range(0, 2) ] for j in range(0, self.synths_count)]
        # we do not retrieve the headers' line. Header is :
        #   synth_id  ;  from_interpolation  ;  parameter  ;  time  ;  value
        self.raw_data = raw_data[1:raw_data.shape[0]-1, ...]
        # decomposition of the main data matrix into several matrices, each
        # representing a cycle of the experiment
        last_synth_id = -1000  # for detecting a new cycle in the main matrix
        last_from_interp = -1000  # idem
        last_param_id = -1000  # for detecting a new parameter (they are stored one after the other)
        last_sub_matrix_first_row = -1  # unvalid value
        # the whole matrix must be read, line by line
        for i in range(0, self.raw_data.shape[0]):
            # new values stored for each new line
            new_synth_id = int(round(self.raw_data[i, 0])) + global_params.synth_id_offset
            new_from_interp = int(round(self.raw_data[i, 1]))
            new_param_id = int(round(self.raw_data[i, 2])) - 1  # params ids start from 1 in the CSV file
            # detection of a new submatrix, if anything changed
            if (last_synth_id != new_synth_id) or (last_from_interp != new_from_interp) or (last_param_id != new_param_id):
                # copy of the last sub-matrix
                if last_sub_matrix_first_row >= 0:  # we skip the very first new block detected...
                    self.data[last_synth_id][last_from_interp][last_param_id] \
                        = self.raw_data[last_sub_matrix_first_row:i, [3, 4]]  # i bound is excluded
                # and we start reading the next one
                last_from_interp = new_from_interp
                last_synth_id = new_synth_id
                last_param_id = new_param_id
                last_sub_matrix_first_row = i
        # processing of the last block (its end is not detected within the for-loop)
        self.data[last_synth_id][last_from_interp][last_param_id]\
            = self.raw_data[last_sub_matrix_first_row:self.raw_data.shape[0], [3, 4]]

        # - - - - - - - - Pre-processing of loaded data : 3 steps - - - - - - - -
        # Warning: for loops do not work by reference.... (nested lists are copied)
        for j in range(0, len(self.data)):
            for j2 in range(0, len(self.data[j])):
                if self.is_cycle_valid[j, j2]:
                    final_time = max([ param_data[:, 0].max() for param_data in self.data[j][j2] ])
                    for k in range(0, len(self.data[j][j2])):
                        # Step 1 : curves closing : addition of values on t=0 and final time (to be searched for)
                        self.data[j][j2][k] = np.vstack( ([0, self.data[j][j2][k][0, 1]],
                                                          self.data[j][j2][k]) )
                        self.data[j][j2][k] = np.vstack( (self.data[j][j2][k],
                                                          [final_time, self.data[j][j2][k][self.data[j][j2][k].shape[0]-1, 1]] ) )
                        # Step 2 : time conversion from milliseconds to seconds
                        self.data[j][j2][k][:, 0] *= 0.001
                        # Step 3 : centering of each parameter curve around its target value
                        # (the values then become the error), and errors normalization
                        self.data[j][j2][k][:, 1] -= synths[j].target_values[k]  # direct centering at first
                        samples_count = self.data[j][j2][k].shape[0] # warning : dont use again if data sizes change again
                        normalization_matrix = np.hstack( (np.ones((samples_count, 1)),
                                                           np.full((samples_count, 1), 1.0/(synths[j].max_values[k] - synths[j].min_values[k]))) )
                        # normalization matrix made for working with "Hadamard" term-by-term matrix multiplication
                        # -> this product is the default with numpy arrays (numpy matrices are not actually common...)
                        self.data[j][j2][k] *= normalization_matrix

        # - - - - - - - - Processing of loaded data : 3 steps - - - - - - - -
        # -> Step 1: performance measured at the end of each cycle. Errors are normalized.
        # pre-allocation of e, d, and s lists - will be 1,5 D lists (synth, interp)
        # (e = error samples, d = duration samples, s = perf score samples)
        self.e_norm1 = np.full((self.synths_count, self.search_types_count), -1.0)
        self.e_norm2 = np.full((self.synths_count, self.search_types_count), -1.0)
        self.d = np.full((self.synths_count, self.search_types_count), -global_params.allowed_time)
        self.s_ingame = np.full((self.synths_count, self.search_types_count), -1.0)
        for j in range(0, len(self.data)):
            for j2 in range(0, len(self.data[j])):
                if self.is_cycle_valid[j, j2]:
                    self.d[j, j2] = self.data[j][j2][0][-1, 0]  # last recorded time of 1st param
                    self.e_norm1[j, j2] = sum([abs(param_data[-1, 1]) for param_data in self.data[j][j2]]) \
                                          / global_params.params_count
                    self.e_norm2[j, j2] = sum( [param_data[-1, 1] ** 2 for param_data in self.data[j][j2]] )
                    self.e_norm2[j, j2] = np.sqrt(self.e_norm2[j, j2]) / global_params.params_count
                    self.s_ingame[j, j2] = perfeval.expe4_ingame_eval(self.e_norm1[j, j2], self.d[j, j2],
                                                                      global_params.allowed_time)
                    # other performance scores will be computed on-demand, after data loading
        # -> Step 2: Mean performance for this subject over all synth sounds, for fader and for interpolation
        # (for quick early results, before a proper analysis)
        self.mean_s_ingame = [-1.0 for j2 in range(self.search_types_count)]
        for j2 in range(self.search_types_count):
            self.mean_s_ingame[j2] = statistics.mean([s for s in self.s_ingame[:, j2] if (s >= 0.0)])
        # -> Step 3: data to be precomputed later
        self.s_adjusted = None

    def precompute_adjusted_s(self):
        """ After construction or unpickling of the class, this method must be called to refresh
        the computation of all possible adjusted performance scores. """
        self.s_adjusted = list()
        for eval_type in perfeval.EvalType:
            if eval_type != perfeval.EvalType.COUNT:
                self.s_adjusted.append(self._compute_s_adjusted(eval_type))

    def get_s_adjusted(self, adjustment_type):
        if self.s_adjusted is None:
            raise RuntimeError("Adjusted performances are not pre-computed yet.")
        if 0 <= int(adjustment_type) < int(perfeval.EvalType.COUNT):
            return self.s_adjusted[int(adjustment_type)]
        else:
            raise ValueError("Requested adjustment function type is not valid.")

    def _compute_s_adjusted(self, adjustment_type):
        s_adj = np.full((self.synths_count, self.search_types_count), -1.0)
        for j in range(0, len(self.data)):
            for j2 in range(0, len(self.data[j])):
                if self.is_cycle_valid[j, j2]:
                    error_type = perfeval.get_error_type_for_adjustment(adjustment_type)
                    if error_type == 1:
                        error_adjusted = self.e_norm1[j, j2]
                    elif error_type == 2:
                        error_adjusted = self.e_norm2[j, j2]
                    s_adj[j, j2] = perfeval.adjusted_eval(error_adjusted, self.d[j, j2],
                                                          self.global_params.allowed_time, adjustment_type)
        return s_adj

    def get_mean_s_adjusted(self, adjustment_type):
        s_adj = self.get_s_adjusted(adjustment_type)
        mean_s = [-1.0 for j2 in range(self.search_types_count)]
        for j2 in range(self.search_types_count):
            mean_s[j2] = statistics.mean([s for s in s_adj[:, j2] if (s >= 0.0)])
        return mean_s

    def get_actual_s_adjusted_1d(self, adjustement_type):
        """ Flattened array of all adjusted perfs, unsorted, not including trial or unvalid data. """
        s_notrial = self.get_s_adjusted(adjustement_type)[self.global_params.synths_trial_count:, :]
        s_notrial = s_notrial.flatten()
        return s_notrial[s_notrial >= 0.0]

    def get_all_s_dataframe(self, eval_type):
        """ Returns a dataframe containing all valid perfs, for a given perf evaluation. Columns are:
        subject_index, expertise_level, synth_id, search_type, performance """
        # only non-trial synths (still might contain unvalid perfs)
        s_notrial = self.get_s_adjusted(eval_type)[self.global_params.synths_trial_count:, :]
        perfs_list = list()
        for j in range(0, s_notrial.shape[0]):
            for j2 in range(0, s_notrial.shape[1]):
                # negative unvalid perfs might remain at this point
                if s_notrial[j, j2] >= 0.0:
                    perfs_list.append([self.index, self.expertise_level, j,
                                      MethodType(j2).name.lower(), s_notrial[j, j2]])
        return pd.DataFrame(perfs_list, columns=['subject_index', 'expertise_level', 'synth_id',
                                                 'search_type', 'performance'])



    @ property
    def synths_count(self):
        return self.global_params.synths_count

    @ property
    def search_types_count(self):
        return self.global_params.search_types_count

    @ property
    def cycles_count(self):
        return self.global_params.cycles_count

    @ property
    def synth_id_offset(self):
        return self.global_params.synth_id_offset

    @ property
    def params_count(self):
        return self.global_params.params_count

    def set_index(self, new_index):
        self.index = new_index


class MethodsOpinion:
        """
        Stores the final opinion on control methods from the XML node. See MethodType for the base enum type
        """
        def __init__(self, opinion_et):
            # COULD BE OPTIMIZED USING PANDAS
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
                return MethodType.SLIDERS
            else:
                return MethodType.NONE
