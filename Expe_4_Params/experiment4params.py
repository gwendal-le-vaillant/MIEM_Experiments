
import experimentdataprocessing as edp
import experimentdatadisplay as edd


# Chargement des données
expe4 = edp.load_experiment_once_a_day("./expe_data/")

# Affichage des basiques de l'expérience : sujets, fonction de score, ...
edd.display_performance_score_surface(params_count=4)
#edd.display_age_and_sex(expe4)

# Compute/extract all interesting data about performances

pass

