
import matplotlib.pyplot as plt
import numpy as np


import experimentdataprocessing as edp
import experimentdatadisplay as edd
import perfeval


# Data loading and pre-processing
expe4 = edp.load_experiment_once_a_day("./expe_data/", force_reload=False)
print(expe4)


# ----- 0 - Display of generic info -----
#edd.plot_age_and_sex(expe4)


# ----- 1 - Display of one subject performance (in interactive plot windows) -----
subject_index = -1  # -1 to disable per-subject data visualization
if (subject_index >= 0) and (subject_index < len(expe4.subjects)):
    subject_curves_visualizer = edd.SubjectCurvesVisualizer(expe4, subject_index,
                                                            show_radio_selector=True)
    subject_perf_visualizer = edd.SubjectPerformancesVisualizer(expe4, subject_index,
                                                                show_radio_selector=True)
# 1 bis Save all curve/perf figures to PDF files
#edd.save_all_subjects_to_pdf(expe4)


# ----- 2 - Analysis of performance evaluation function, Display of global performance results -----
perfs_analyzer = perfeval.Analyzer(expe4)
perfs_analyzer.compare_ingame_to_adjusted()
perfs_analyzer.compare_adjusted()
#edd.plot_all_perfs(expe4)
#edd.plot_perf_and_expertise(expe4)


# ----- 3 - Others -----
#edd.plot_opinions_on_methods(expe4)


# Show all previously built figures
plt.show()
