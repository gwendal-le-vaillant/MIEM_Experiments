
import matplotlib.pyplot as plt

import experimentdataprocessing as edp
import experimentdatadisplay as edd


# Data loading and pre-processing
expe4 = edp.load_experiment_once_a_day("./expe_data/", force_reload=False)
print(expe4)

# 0 - Display of generic info
#edd.display_performance_score_surface(params_count=4)
#edd.display_age_and_sex(expe4)

# 1 - Display of one subject performance (in interactive plot windows)
subject_index = 21  # -1 to disable per-subject data visualization
if (subject_index >= 0) and (subject_index < len(expe4.subjects)):
    subject_curves_visualizer = edd.SubjectCurvesVisualizer(expe4, subject_index,
                                                            show_radio_selector=True)
    subject_perf_visualizer = edd.SubjectPerformancesVisualizer(expe4,
                                                                subject_index,
                                                                show_radio_selector=True)

# 2 - Display of global performance results

# Show all previously built figures
plt.show()
