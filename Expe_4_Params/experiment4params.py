
import matplotlib.pyplot as plt
import numpy as np

import experimentdataprocessing as edp
import experimentdatadisplay as edd
import perfeval

# ----- 0 - Data loading and pre-processing + Display of generic info -----
expe4 = edp.load_experiment_once_a_day("./expe_data/", force_reload=False)
print(expe4)
expe4.precompute_adjusted_s()
# edd.plot_age_and_sex(expe4)

# ----- 1 - Display of one subject performance (in interactive plot windows) -----
subject_index = -1  # -1 to disable per-subject data visualization
if (subject_index >= 0) and (subject_index < len(expe4.subjects)):
    subject_curves_visualizer = edd.SubjectCurvesVisualizer(expe4, subject_index, show_radio_selector=True)
    subject_perf_visualizer = edd.SubjectPerformancesVisualizer(expe4, subject_index, show_radio_selector=True)
# 1 bis Save all curve/perf figures to PDF files
# edd.save_all_subjects_to_pdf(expe4)

# ----- 2 - Analysis of performance evaluation functions -----
perfs_analyzer = perfeval.Analyzer(expe4)
# perfs_analyzer.plot_adjusted_perf_only()
# perfs_analyzer.compare_adjusted()
# perfs_analyzer.compare_adjusted(adj_types=[perfeval.EvalType.ADJUSTED, perfeval.EvalType.FOCUS_ON_ERROR, perfeval.EvalType.FOCUS_ON_TIME])
# perfs_analyzer.compare_adjusted(adj_types=[perfeval.EvalType.ALPHA_EXPE, perfeval.EvalType.INGAME, perfeval.EvalType.ADJUSTED])

# ----- 3 - Display of global performance results -----
# ----- ----- 3 a) Sorted by method only (p-values not displayed, always very close to 0.0)
# edd.all_perfs_histogram(expe4, display_KS=False)
# edd.all_perfs_histogram(expe4, perf_eval_type=perfeval.EvalType.INGAME)
# edd.all_perfs_histogram(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_ERROR)
# edd.all_perfs_histogram(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_TIME)
# ----- ----- 3 b) Sorted by method and by synth ID
# edd.plot_all_perfs_per_synth(expe4)
# edd.plot_all_perfs_per_synth(expe4, perf_eval_type=perfeval.EvalType.INGAME)
# edd.plot_all_perfs_per_synth(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_ERROR)
# edd.plot_all_perfs_per_synth(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_TIME)
# edd.plot_all_perfs_histograms_by_synth(expe4, display_tests=True)  # also displays Wilcoxon signed rank tests
# ----- ----- 3 c) Sorted by method and expertise level (boxplots)
# edd.plot_all_perfs_by_expertise(expe4, perfeval.EvalType.ADJUSTED, add_swarm_plot=True)
# edd.plot_all_perfs_by_expertise(expe4, perfeval.EvalType.INGAME)
# edd.plot_all_perfs_by_expertise(expe4, perfeval.EvalType.FOCUS_ON_TIME)
# edd.plot_all_perfs_by_expertise(expe4, perfeval.EvalType.FOCUS_ON_ERROR)
# ----- ----- 3 d) Average perfs of subjects, sorted by method and expertise level
edd.fit_perf_vs_expertise(expe4, perf_eval_type=perfeval.EvalType.ADJUSTED, show_fit_analysis=False)
# edd.fit_perf_vs_expertise(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_TIME)
# edd.fit_perf_vs_expertise(expe4, perf_eval_type=perfeval.EvalType.FOCUS_ON_ERROR)
# edd.fit_perf_vs_expertise(expe4, perf_eval_type=perfeval.EvalType.INGAME)

# ----- 4 - Others -----
# edd.plot_opinions_on_methods(expe4)

# ----- 5 - Show all previously built figures -----
plt.show()
