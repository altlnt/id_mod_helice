# Identification de modèles 

Les scripts sont numérotés, dans l'ordre d'utilisation.

- On commence par parser les logs avec 0_0_log_class.py

- On les traite avec 0_5_process_log_real

- On utilise ensuite les scripts id

    * 1_id_simple_nosympy_corrected.py puis 2_id_helices_parallel.py pour le modèle hélice
    * 2_id_avion_parallel pour le modèle avion
    
- On agrege les résultats pour chaque optimisation avec 3_0_agregate_results.ipynb

- On agrege le tout, dans un répertoire de csvs, et on fait le bilan des optis


