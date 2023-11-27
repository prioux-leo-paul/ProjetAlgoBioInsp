import matplotlib.pyplot as plt
import pandas as pd

def getMeanByDimension(bestFitness,dimension,nbProblem):
    '''
    La fonction fais la moyenne des execution et 
    la moyenne de toute les fonctions d'évaluation par dimension pour un seul dataframe
    input : bestFitness (un dataframe), dimension(une liste de dimmension), nbExec (int du nombre d'éxecution), nbProblem (int du nombre de problème) 
    output : dataframe
    '''
    bestFitness=bestFitness.stack(level=0).unstack()
    df = pd.DataFrame(columns=['F'+str(x+1) for x in range(nbProblem)],index=['Dim_'+str(x) for x in dimension])
    for dim in dimension:
        df2 = pd.DataFrame(columns=["F"+str(x+1) for x in range(nbProblem)])
        for i in range(nbProblem):
            df2["F"+str(i+1)]= bestFitness['F'+str(i+1)+'_'+str(dim)+'D'].min(axis=0)
        df.loc['Dim_'+str(dim)] = df2.min(axis=0)
    return df

def getMeanByDimensionForOptimal(bestFitness,dimension,nbProblem):
    '''
    La fonction fais la moyenne des execution et 
    la moyenne de toute les fonctions d'évaluation par dimension pour un seul dataframe
    input : bestFitness (un dataframe), dimension(une liste de dimmension), nbExec (int du nombre d'éxecution), nbProblem (int du nombre de problème) 
    output : dataframe
    '''
    df = pd.DataFrame(columns=['F'+str(x+1) for x in range(nbProblem)],index=['Dim_'+str(x) for x in dimension])
    for dim in dimension:
        df2 = pd.DataFrame(index=[0],columns=["F"+str(x+1) for x in range(nbProblem)])
        for i in range(nbProblem):
            df2["F"+str(i+1)]= bestFitness.loc[0,'F'+str(i+1)+'_'+str(dim)+'D'].tolist()
        df.loc['Dim_'+str(dim)] = df2.loc[0].tolist()
    return df

def getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessGDE3,bestFitnessSMPSO,bestFitnessOptimal):
    '''
    La fonction fais la moyenne des execution et 
    la moyenne de toute les fonctions d'évaluation par dimension pour trois dataframes 
    et les rassemble par dimension
    input : bestFitnessGA (dataframe), bestFitnessSMPSO (dataframe), bestFitnessGDE3 (dataframe)
    output : 4 dataframe
    '''

    dimension = [2,10,30,50]
    dfGA=getMeanByDimension(bestFitnessGA,dimension,25)
    dfSMPSO=getMeanByDimension(bestFitnessSMPSO,dimension,25)
    dfGDE3=getMeanByDimension(bestFitnessGDE3,dimension,25)
    dfOptimal = getMeanByDimensionForOptimal(bestFitnessOptimal,[2,10,30,50],25)
    dfPlotDim_2 = pd.DataFrame(columns=[x+1 for x in range(25)],index=['GA','SMPSO','GDE3','Optimal'])
    dfPlotDim_10 = pd.DataFrame(columns=[x+1 for x in range(25)],index=['GA','SMPSO','GDE3','Optimal'])
    dfPlotDim_30 = pd.DataFrame(columns=[x+1 for x in range(25)],index=['GA','SMPSO','GDE3','Optimal'])
    dfPlotDim_50 = pd.DataFrame(columns=[x+1 for x in range(25)],index=['GA','SMPSO','GDE3','Optimal'])

    for i in dimension:
        locals()['dfPlotDim_'+str(i)].loc['GA'] = dfGA.loc['Dim_'+str(i)].to_list()
        locals()['dfPlotDim_'+str(i)].loc['GDE3'] = dfGDE3.loc['Dim_'+str(i)].to_list()
        locals()['dfPlotDim_'+str(i)].loc['SMPSO'] = dfSMPSO.loc['Dim_'+str(i)].to_list()
        locals()['dfPlotDim_'+str(i)].loc['Optimal'] = dfOptimal.loc['Dim_'+str(i)].to_list()

    
    return dfPlotDim_2,dfPlotDim_10,dfPlotDim_30,dfPlotDim_50

def plotMeanXoversMutationsbyEvaluationFunction(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3,bestFitnessOptimal):
    '''
    Créer un plot de la moyenne de toute les fonctions d'évaluations des trois dataframe pour chaque dimension
    input : bestFitnessGA (dataframe), bestFitnessSMPSO (dataframe), bestFitnessGDE3 (dataframe)
    output : affiche un plot
    '''
    dfPlotDim_2,dfPlotDim_10,dfPlotDim_30,dfPlotDim_50 = getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3,bestFitnessOptimal)
    tmpsuccesRateDim_2 = pd.DataFrame(index=['GA','SMPSO','GDE3'],columns=[x+1 for x in range(25)])
    tmpsuccesRateDim_10 = pd.DataFrame(index=['GA','SMPSO','GDE3'],columns=[x+1 for x in range(25)])
    tmpsuccesRateDim_30 = pd.DataFrame(index=['GA','SMPSO','GDE3'],columns=[x+1 for x in range(25)])
    tmpsuccesRateDim_50 = pd.DataFrame(index=['GA','SMPSO','GDE3'],columns=[x+1 for x in range(25)])
    for dim in [2,10,30,50]:
        for i in range(25):
            locals()['tmpsuccesRateDim_'+str(dim)].loc['GA',i+1] = True if locals()['dfPlotDim_'+str(dim)].loc['GA',i+1] < locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1] + locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1]*0.05 else False 
            locals()['tmpsuccesRateDim_'+str(dim)].loc['GDE3',i+1] = True if locals()['dfPlotDim_'+str(dim)].loc['GDE3',i+1] < locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1] + locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1]*0.05 else False 
            locals()['tmpsuccesRateDim_'+str(dim)].loc['SMPSO',i+1] = True if locals()['dfPlotDim_'+str(dim)].loc['SMPSO',i+1] < locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1] + locals()['dfPlotDim_'+str(dim)].loc['Optimal',i +1]*0.05 else False 

    successRate = pd.DataFrame(index=['GA','GDE3','SMPSO'],columns=['D_2','D_10','D_30','D_50'])
    for dim in [2,10,30,50]:
        successRate.loc['GA','D_'+str(dim)]  = sum(locals()['tmpsuccesRateDim_'+str(dim)].loc['GA'].tolist()) / 25
        successRate.loc['GDE3','D_'+str(dim)] = sum(locals()['tmpsuccesRateDim_'+str(dim)].loc['GDE3'].tolist()) / 25
        successRate.loc['SMPSO','D_'+str(dim)] = sum(locals()['tmpsuccesRateDim_'+str(dim)].loc['SMPSO'].tolist()) / 25
    
    result = []
    techniques = ["GDE3","SMPSO","GA"]
    for technique in techniques :
        result.append(build_method_comparaison_dimension(technique))

    reverse_list_function = [list(x) for x  in zip(result[0],result[1],result[2])]
    
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('best-values')
    ax1.plot(reverse_list_function)
    ax1.tick_params(axis='y')
    
    plt.legend(techniques, title="best-values",loc="center left" )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Succes Rate')  # we already handled the x-label with ax1
    ax2.plot(successRate.transpose(), linestyle='dashed')
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.legend(techniques,title="Succes Rate",loc="center right")
  
    plt.title("Evolution best values en fonction de la dimension")
    
    plt.show()
    
    



def get_df(technique) : 
    '''
    input  : technique voulu (String)
    output : dataframe correspondant
    
    Charge le fichier correspondant à la technique et retourne son df
    '''
    return  pd.read_csv (r'ProjetAlgoBioInsp/%s'%technique+'_experiment_10_runs_dim_2_10_30_50.csv'  ,header=[0,1])


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
    return list_function_medianed
       
    
def main():
    bestFitnessGA = pd.read_csv (r'ProjetAlgoBioInsp/GA_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessSMPSO = pd.read_csv (r'ProjetAlgoBioInsp/SMPSO_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessGDE3 = pd.read_csv (r'ProjetAlgoBioInsp/GDE3_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessOptimal = pd.read_csv(r'ProjetAlgoBioInsp/Optimal_Value.csv')
    bestFitnessOptimal.columns = ["name","value"]
    bestFitnessOptimal = bestFitnessOptimal.set_index("name")
    bestFitnessOptimal = bestFitnessOptimal.transpose()
    bestFitnessOptimal = pd.DataFrame(bestFitnessOptimal.loc['value'].tolist(),index=bestFitnessOptimal.columns.tolist()).transpose()

    plotMeanXoversMutationsbyEvaluationFunction(bestFitnessGA,bestFitnessGDE3,bestFitnessSMPSO,bestFitnessOptimal)

if __name__ == '__main__':
    main()