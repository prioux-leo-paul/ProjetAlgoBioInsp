
'''
Methode :
    Moyenne des Xovers et mutation pour chaque fonction d'evaluation ( 4 plot, separer par dimensions)
    Moyenne de fonction d evaluation ( pour Xover et mutation ) par technique 
    Répartition des bestvalues de chaque fonctions
    Meilleur methode par fonction 
    Comparaison methode en fonction des dimensions
'''

'''
Conclusion Possible : 
    Toute méthode a bcp de mal à opti pour dim 30 et 50
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# commune à tout les graphes
def get_df(technique) : 
    '''
    input  : technique voulu (String)
    output : dataframe correspondant
    
    Charge le fichier correspondant à la technique et retourne son df
    '''
    return  pd.read_csv(r'./%s'%technique+'_experiment_10_runs_dim_2_10_30_50.csv'  ,header=[0,1])


#méthode 1 
def median_and_clean_all_values_df(technique) :
    '''
    input  : technique (String)
    output : list
    Retourne la moyenne des differents itérations des best values de la techniques 
    
    '''
    df_technique = get_df(technique)
    df_technique.columns = range(df_technique.shape[1])   # Delete headers
    df_technique_median = df_technique.median() 
    df_technique_median_list = df_technique_median.tolist()
    return df_technique_median_list
    
def build_and_plot_repartion_bestvalues(techniques) : 
    '''
    input  : liste des techniques (list:String)
    output : plot
    Itère sur les techniques données en argument et renvoie une comparaison des différentes répartitions des techniques
    (diagramme en boite à moustache) 
    
    '''
    median_all_df = []
    for technique in techniques : 
        median_all_df.append(median_and_clean_all_values_df(technique))
    #print(median_all_df)
    
    print(np.var(median_all_df[0]))
    print(np.var(median_all_df[1]))
    print(np.var(median_all_df[2]))
    
    # plot 
    plt.boxplot(median_all_df,labels=techniques)
    plt.title("Répartition des best values de chaque fonction d'évalution")
    plt.ylabel("log best values")
    plt.yscale("log")
    plt.show()
    
#méthode 2

def build_best_method_for_a_function(technique) :
    '''
    input  : liste des techniques (list:String)
    output : plot
    Itère sur les techniques données en argument et renvoie une comparaison des différentes répartitions des techniques
    (diagramme en boite à moustache) 
    
    '''
    df_technique = get_df(technique)
   
    df_technique=df_technique.stack(level=0).unstack()
    df_by_function_and_dim =  df_technique.groupby(level=0).agg(['median']).stack(1)
    df_by_function_and_dim_medianed = df_by_function_and_dim.median(axis = 0)
    list_function_medianed = []
    col_name = []
    for fonction in range(1,26) :
        for dim in [2,10,30,50] :
            col_name.append('F%d'%fonction+'_'+"%d"%dim+'D')
        list_function_medianed.append(df_by_function_and_dim_medianed[col_name].median())
    
    return list_function_medianed
       
def plot_best_method_for_a_function(techniques) :
    '''
    input  : liste des techniques (list:String)
    output : barplot 
    Itère sur les techniques données en argument et renvoie une comparaison des performances des différentes méthodes
    pour chaque fonction. 
    '''
    result = []
    for technique in techniques :
        result.append(build_best_method_for_a_function(technique))
    print(result)
    
    # inversion des dimensions de la liste 
    reverse_list_function = [list(x) for x  in zip(result[0],result[1],result[2])]
    print(reverse_list_function)
    
    # plot 
    plt.plot(reverse_list_function)
    plt.title("Meilleur methode par fonction (aggregation par mediane)")
    #plt.ylim(-500,18000)
    plt.xlabel("f-evals")
    plt.ylabel("log best values")
    plt.legend(techniques)
    plt.yscale('log')
    plt.show()
    
    
#méthode 3

def build_method_comparaison_dimension(technique) :
    '''
    input  : technique (String)
    output : list
    Retourne une liste de la moyenne des itérations(n-seeds) de best values par function 
    '''
    
    df_technique = get_df(technique)
   
    df_technique=df_technique.stack(level=0).unstack()
    df_by_function_and_dim =  df_technique.groupby(level=0).agg(['median']).stack(1)
    df_by_function_and_dim_medianed = df_by_function_and_dim.median(axis = 0)
    list_function_medianed = []
    col_name = []
    for dim in [2,10,30,50] :
        for fonction in range(1,26) :
            col_name.append('F%d'%fonction+'_'+"%d"%dim+'D')
        list_function_medianed.append(df_by_function_and_dim_medianed[col_name].median())
    print(list_function_medianed)
    return list_function_medianed
       
def plot_method_comparaison_dimension(techniques) :
    '''
    input  : techniques (list:String)
    output : pltot
    Récupere une list des moyennes des bestvalues en fonction des fonctions de teste
    et inverse les dimensions de la list à fin d'obtenir la moyenne en fonction des dimensions.
    Plot une graph montrant l'évolution de la mean(best value) en fonction des dimensions
    '''
    result = []
    
    for technique in techniques :
        result.append(build_method_comparaison_dimension(technique))

    
    # inversion des dimensions de la liste 
    reverse_list_function = [list(x) for x  in zip(result[0],result[1],result[2])]
    print(reverse_list_function)
    
    # plot 
    plt.plot(reverse_list_function)
    plt.title("Evolution best values en fonction de la dimension")
    plt.xlabel("Dimensions")
    plt.ylabel("best values")
    plt.legend(techniques)
    plt.show()
    

#main (execution)

if __name__ == "__main__" :
    techniques = ["GDE3","SMPSO","GA"]
    build_and_plot_repartion_bestvalues(techniques)
    plot_best_method_for_a_function(techniques)
    plot_method_comparaison_dimension(techniques) 