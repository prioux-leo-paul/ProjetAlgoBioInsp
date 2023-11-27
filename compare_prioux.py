import pandas as pd
import matplotlib.pyplot as plt
from Genetical_Algorithm import bestFitness
from scipy import stats
from platypus import *




def getMeanByDimension(bestFitness,dimension,nbExec,nbProblem):
    df = pd.DataFrame(columns=[x+1 for x in range(nbExec)],index=['Dim_'+str(x) for x in dimension])
    for dim in dimension:
        df2 = pd.DataFrame(columns=["c_"+str(x+1) for x in range(nbProblem)])
        for i in range(nbProblem):
            df2["c_"+str(i+1)]= bestFitness['F'+str(i+1)+'_'+str(dim)+'D'].median(axis=1)
        df.loc['Dim_'+str(dim)] = df2.median(axis=1)
    return df

def getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3):
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
    bestFitnessGDE3=bestFitnessGDE3.stack(level=0).unstack()
    bestFitnessGA=bestFitnessGA.stack(level=0).unstack()
    bestFitnessSMPSO=bestFitnessSMPSO.stack(level=0).unstack()

    dfPlotDim_2,dfPlotDim_10,dfPlotDim_30,dfPlotDim_50 = getMeanByDimensionByTechnique(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Moyenne des Xovers et mutation pour chaque fonction d\'evaluation')
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
        ax.set(xlabel='Execution', ylabel='f-evals')
    plt.show()


def plotComparaisonTechnicalForEachXoverMutation(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3):
    bestFitnessGDE3=bestFitnessGDE3.stack(level=0).unstack()
    bestFitnessGA=bestFitnessGA.stack(level=0).unstack()
    bestFitnessSMPSO=bestFitnessSMPSO.stack(level=0).unstack()   
    Df_SBX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    Df_SBX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    Df_SPX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    Df_SPX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['GA','SMPSO','GDE3'])
    for n in ['GA','SMPSO','GDE3']:
        fTmpDf_SBX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['median'])
        fTmpDf_SBX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['median'])
        fTmpDf_SPX_PM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['median'])
        fTmpDf_SPX_UM = pd.DataFrame(columns=[str(x+1) for x in range(25) ],index=['median'])
        for j in range(25):
            listTmp = []
            for i in [2,10,30,50]:
                listTmp.append('F'+str(j+1)+'_'+str(i)+'D')
            tmpDf_SBX_PM = pd.DataFrame(columns=listTmp,index=['median'])
            tmpDf_SBX_UM = pd.DataFrame(columns=listTmp,index=['median'])
            tmpDf_SPX_PM = pd.DataFrame(columns=listTmp,index=['median'])
            tmpDf_SPX_UM = pd.DataFrame(columns=listTmp,index=['median'])

            for i in listTmp:
                #locals()['bestFitness'+str(n)]
                tmpDf_SBX_PM[i] = locals()['bestFitness'+str(n)][i]['GA_SBX_PM'].median(axis=0)
                tmpDf_SBX_UM[i] =  locals()['bestFitness'+str(n)][i]['GA_SBX_UM'].median(axis=0)
                tmpDf_SPX_PM[i] =  locals()['bestFitness'+str(n)][i]['GA_SPX_PM'].median(axis=0)
                tmpDf_SPX_UM[i] =  locals()['bestFitness'+str(n)][i]['GA_SPX_UM'].median(axis=0)
            
            fTmpDf_SBX_PM[str(j+1)] = tmpDf_SBX_PM.median(axis=1)
            fTmpDf_SBX_UM[str(j+1)] = tmpDf_SBX_UM.median(axis=1)
            fTmpDf_SPX_PM[str(j+1)] = tmpDf_SPX_PM.median(axis=1)
            fTmpDf_SPX_UM[str(j+1)] = tmpDf_SPX_UM.median(axis=1)
        Df_SBX_PM.loc[n] = fTmpDf_SBX_PM.loc['median']
        Df_SBX_UM.loc[n] = fTmpDf_SBX_UM.loc['median']
        Df_SPX_PM.loc[n] = fTmpDf_SPX_PM.loc['median']
        Df_SPX_UM.loc[n] = fTmpDf_SPX_UM.loc['median']

    
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
    for ax in axs.flat:
        ax.set(xlabel='Fonction d\'évaluation', ylabel='f-evals')
    plt.show()


if __name__ == '__main__':
    # load from saved file
    bestFitnessGA = pd.read_csv (r'ProjetAlgoBioInsp/GA_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessSMPSO = pd.read_csv (r'ProjetAlgoBioInsp/SMPSO_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])
    bestFitnessGDE3 = pd.read_csv (r'ProjetAlgoBioInsp/GDE3_experiment_10_runs_dim_2_10_30_50.csv',header=[0,1])

    #plotMeanXoversMutationsbyEvaluationFunction(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
    plotComparaisonTechnicalForEachXoverMutation(bestFitnessGA,bestFitnessSMPSO,bestFitnessGDE3)
