import pandas as pd
import matplotlib.pyplot as plt
from Genetical_Algorithm import bestFitness
from scipy import stats
from platypus import *

'''
Methode :
    Moyenne des Xovers et mutation pour chaque fonction d'evaluation ( 4 plot, separer par dimensions)
    Moyenne de fonction d evaluation ( pour Xover et mutation ) par technique 
    Répartition des bestvalues de chaque fonctions
    Meilleur methode par fonction 
    Comparaison methode en fonction des dimensions
'''

def getMeanByDimension(bestFitness,dimension,nbExec,nbProblem):
    '''
    La fonction fais la moyenne des execution et 
    la moyenne de toute les fonctions d'évaluation par dimension pour un seul dataframe
    input : bestFitness (un dataframe), dimension(une liste de dimmension), nbExec (int du nombre d'éxecution), nbProblem (int du nombre de problème) 
    output : dataframe
    '''
    df = pd.DataFrame(columns=[x+1 for x in range(nbExec)],index=['Dim_'+str(x) for x in dimension])
    for dim in dimension:
        df2 = pd.DataFrame(columns=["c_"+str(x+1) for x in range(nbProblem)])
        for i in range(nbProblem):
            df2["c_"+str(i+1)]= bestFitness['F'+str(i+1)+'_'+str(dim)+'D'].mean(axis=1)
        df.loc['Dim_'+str(dim)] = df2.mean(axis=1)
    return df

def getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3):
    '''
    La fonction fais la moyenne des execution et 
    la moyenne de toute les fonctions d'évaluation par dimension pour trois dataframes 
    et les rassemble par dimension
    input : bestFitnessGA (dataframe), bestFitnessSMPSO (dataframe), bestFitnessGDE3 (dataframe)
    output : 4 dataframe
    '''
    dimension = [2,10,30,50]
    dfGA=getMeanByDimension(bestFitnessGA,dimension,9,25)
    dfSMPSO=getMeanByDimension(bestFitnessSMPSO,dimension,9,25)
    dfGDE3=getMeanByDimension(bestFitnessGDE3,dimension,9,25)
    dfPlotDim_2 = pd.DataFrame(columns=[x+1 for x in range(9)],index=['GA','SMPSO','GDE3'])
    dfPlotDim_10 = pd.DataFrame(columns=[x+1 for x in range(9)],index=['GA','SMPSO','GDE3'])
    dfPlotDim_30 = pd.DataFrame(columns=[x+1 for x in range(9)],index=['GA','SMPSO','GDE3'])
    dfPlotDim_50 = pd.DataFrame(columns=[x+1 for x in range(9)],index=['GA','SMPSO','GDE3'])

    for i in dimension:
        locals()['dfPlotDim_'+str(i)].loc['GA'] = dfGA.loc['Dim_'+str(i)].to_list()
        locals()['dfPlotDim_'+str(i)].loc['GDE3'] = dfGDE3.loc['Dim_'+str(i)].to_list()
        locals()['dfPlotDim_'+str(i)].loc['SMPSO'] = dfSMPSO.loc['Dim_'+str(i)].to_list()
    
    return dfPlotDim_2,dfPlotDim_10,dfPlotDim_30,dfPlotDim_50


def plotMeanXoversMutationsbyEvaluationFunction(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3):
    '''
    Créer un plot de la moyenne de toute les fonctions d'évaluations des trois dataframe pour chaque dimension
    input : bestFitnessGA (dataframe), bestFitnessSMPSO (dataframe), bestFitnessGDE3 (dataframe)
    output : affiche un plot
    '''
    bestFitnessGDE3=bestFitnessGDE3.stack(level=0).unstack()
    bestFitnessGA=bestFitnessGA.stack(level=0).unstack()
    bestFitnessSMPSO=bestFitnessSMPSO.stack(level=0).unstack()

    dfPlotDim_2,dfPlotDim_10,dfPlotDim_30,dfPlotDim_50 = getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Moyenne des crossovers et des mutations de toute les fonctions d\'évaluations')
    fig.tight_layout(pad=3.0)
    axs[0, 0].plot(dfPlotDim_2.transpose())
    axs[0, 0].set_title('Dimension 2')
    axs[0, 1].plot(dfPlotDim_10.transpose())
    axs[0, 1].set_title('Dimension 10')
    axs[1, 0].plot(dfPlotDim_30.transpose())
    axs[1, 0].set_title('Dimension 30')
    axs[1, 1].plot(dfPlotDim_50.transpose())
    axs[1, 1].set_title('Dimension 50')
    fig.legend(labels=["GA","SMPSO","GDE3"],   # The labels for each line
           loc="lower right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )
    for ax in axs.flat:
        ax.set(xlabel='Execution', ylabel='Best value')
    plt.show()


def plotComparaisonTechnicalForEachXoverMutation(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3):
    '''
    Créer un plot de la moyenne de toute les fonctions d'évaluations et de toute les dimensions des trois dataframe pour chaque technique Xovers mutations
    input : bestFitnessGA (dataframe), bestFitnessSMPSO (dataframe), bestFitnessGDE3 (dataframe)
    output : affiche un plot
    '''
    bestFitnessGDE3=bestFitnessGDE3.stack(level=0).unstack()
    bestFitnessGA=bestFitnessGA.stack(level=0).unstack()
    bestFitnessSMPSO=bestFitnessSMPSO.stack(level=0).unstack()   
    Df_SBX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    Df_SBX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    Df_SPX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25)],index=['GA','SMPSO','GDE3'])
    Df_SPX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    for n in ['GA','SMPSO','GDE3']:
        fTmpDf_SBX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['mean'])
        fTmpDf_SBX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['mean'])
        fTmpDf_SPX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['mean'])
        fTmpDf_SPX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['mean'])
        for j in range(25):
            listTmp = []
            for i in [2,10,30,50]:
                listTmp.append('F'+str(j+1)+'_'+str(i)+'D')
            tmpDf_SBX_PM = pd.DataFrame(columns=listTmp,index=['mean'])
            tmpDf_SBX_UM = pd.DataFrame(columns=listTmp,index=['mean'])
            tmpDf_SPX_PM = pd.DataFrame(columns=listTmp,index=['mean'])
            tmpDf_SPX_UM = pd.DataFrame(columns=listTmp,index=['mean'])

            for i in listTmp:
                tmpDf_SBX_PM[i] = locals()['bestFitness'+str(n)][i][str(n)+'_SBX_PM'].mean(axis=0)
                tmpDf_SBX_UM[i] =  locals()['bestFitness'+str(n)][i][str(n)+'_SBX_UM'].mean(axis=0)
                tmpDf_SPX_PM[i] =  locals()['bestFitness'+str(n)][i][str(n)+'_SPX_PM'].mean(axis=0)
                tmpDf_SPX_UM[i] =  locals()['bestFitness'+str(n)][i][str(n)+'_SPX_UM'].mean(axis=0)
            
            fTmpDf_SBX_PM[str(j+1)] = tmpDf_SBX_PM.mean(axis=1)
            fTmpDf_SBX_UM[str(j+1)] = tmpDf_SBX_UM.mean(axis=1)
            fTmpDf_SPX_PM[str(j+1)] = tmpDf_SPX_PM.mean(axis=1)
            fTmpDf_SPX_UM[str(j+1)] = tmpDf_SPX_UM.mean(axis=1)
            Df_SBX_PM.loc[n] = fTmpDf_SBX_PM.loc['mean']
            Df_SBX_UM.loc[n] = fTmpDf_SBX_UM.loc['mean']
            Df_SPX_PM.loc[n] = fTmpDf_SPX_PM.loc['mean']
            Df_SPX_UM.loc[n] = fTmpDf_SPX_UM.loc['mean']

    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Comparaison des techniques pour chaque méthode des Xovers et mutations')
    fig.tight_layout(pad=3.0)
    axs[0, 0].plot(Df_SBX_PM.transpose())
    axs[0, 0].set_title('Méthode SBX PM')
    axs[0, 1].plot(Df_SBX_UM.transpose())
    axs[0, 1].set_title('Méthode SBX UM')
    axs[1, 0].plot(Df_SPX_PM.transpose())
    axs[1, 0].set_title('Méthode SPX PM')
    axs[1, 1].plot(Df_SPX_UM.transpose())
    axs[1, 1].set_title('Méthode SPX UM')
    fig.legend(labels=["GA","SMPSO","GDE3"],   # The labels for each line
           loc="lower right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )
    for x in range(2) :
        for y in range(2) :
            axs[x, y].set_yscale('log')
    for ax in axs.flat:
        ax.set(xlabel='Fonction d\'évaluation', ylabel='Log best value')
    plt.show()

# commune à tout les graphes
def get_df(technique) : 
    '''
    input  : technique voulu (String)
    output : dataframe correspondant
    
    Charge le fichier correspondant à la technique et retourne son df
    '''
    return  pd.read_csv (r'ProjetAlgoBioInsp/%s'%technique+'_experiment_10_runs_dim_2_10_30_50.csv'  ,header=[0,1])


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
    print(median_all_df)
    
    # plot 
    plt.boxplot(median_all_df,labels=techniques)
    plt.title("Répartition des best values de chaque technique")
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
    plt.plot(reverse_list_function,)
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
    
    bestFitnessGA = pd.read_csv (r'ProjetAlgoBioInsp/GA_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessSMPSO = pd.read_csv (r'ProjetAlgoBioInsp/SMPSO_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessGDE3 = pd.read_csv (r'ProjetAlgoBioInsp/GDE3_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    plotMeanXoversMutationsbyEvaluationFunction(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
    plotComparaisonTechnicalForEachXoverMutation(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
    