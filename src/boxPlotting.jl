# Questo codice deve partire dagli exp e tirare fuori i parametri
# Li deve organizzare in una matrice per ogni parametro, in cui le colonne
# rappresentano i parametri per ogni paziente per ogni modello


#Settings list
@load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;
a_dist = [exp(sol.u[1]) for sol in best_solution]


boxplot([a_dist, b_dist];
        labels = ["Gruppo A" "Gruppo B"],
        title  = "Box-plot",
        ylabel = "valore")