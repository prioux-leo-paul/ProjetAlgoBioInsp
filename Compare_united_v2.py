#Compare_united_v2.py

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from platypus import *
from Genetical_Algorithm import *
from SMPSO import *
from Differential_Evolution import * 
from scipy.stats import shapiro




# commune à tout les graphes
def get_df(technique) : 
    '''
    input  : technique voulu (String)
    output : dataframe correspondant
    
    Charge le fichier correspondant à la technique et retourne son df
    '''
    return  pd.read_csv(r'ProjetAlgoBioInsp/%s'%technique+'_experiment_10_runs_dim_2_10_30_50.csv'  ,header=[0,1])

def get_fonction_value(df_technique,dimension) :
   
    df_technique=df_technique.stack(level=0).unstack()
    
    
    df_technique_fixed_F_D =df_technique.iloc[:, df_technique.columns.get_level_values(0)== "F1_%d"%dimension+"D" ].mean(axis = 1 )
    
    print("Dimension : %d" %dimension )
    print (df_technique_fixed_F_D)
    
    return list(df_technique_fixed_F_D) 

def get_value_for_a_test(test,technique,dimensions) : 
    df_technique = get_df(technique)
    for x in range(len(dimensions)) : #num of dim
        list_value =  get_fonction_value(df_technique,dimensions[x])
        
        test(list_value) 

def test_pareto(technique) : 
    df = get_df(technique)

#  Normality Tests

# Shapiro-Wilk Test
def Shapiro(x,significance_level=0.05):
    '''Le test de Shapiro-Wilk teste l'hypothèse nulle selon laquelle les données ont été tirées d'une distribution normale.'''
    stat1, p1 = stats.shapiro(x)
    print('stat=%.3f, p=%.3f' % (stat1, p1))
    if p1 > significance_level :
        print('Normally distributed')
    else:
        print('Not Normally distributed')
    return stat1, p1

def Kolmogorov(x,y=None,significance_level=0.05):
    '''
    Le test à un échantillon compare la distribution sous-jacente F(x) d'un échantillon à une distribution donnée G(x). 
    Le test à deux échantillons compare les distributions sous-jacentes de deux échantillons indépendants. 
    Les deux tests ne sont valables que pour les distributions continues.
    '''
    if y == None:
        stat3,p3 =  stats.kstest(x,stats.norm.cdf)
    else:
        stat3,p3 =  stats.kstest(x,y)
    print('stat=%.20f, p=%.3f' % (stat3, p3))
    if p3 > significance_level:
        print('Normally distributed')
    else:
        print('Not Normally distributed')
    return stat3, p3


def Wilcoxon(x,y=None,significance_level=0.05):
    '''
    Le test des rangs signés de Wilcoxon teste l'hypothèse nulle selon laquelle deux échantillons appariés liés proviennent de la même distribution.
    En particulier, il teste si la distribution des différences x - y est symétrique autour de zéro. 
    Il s'agit d'une version non paramétrique du test T apparié.
    '''
    if (y == None):
        stat5,p5 =  stats.wilcoxon(x)
    else:
        stat5,p5 =  stats.wilcoxon(x,y)
    print('stat=%.20f, p=%.3f' % (stat5, p5))
    if p5 > significance_level:
        print('Normally distributed')
    else:
        print('Not Normally distributed')
    return stat5,p5



if __name__ == "__main__" :
    techniques = ["GDE3","SMPSO","GA"]
    dimensions = [2,10,30,50]
    tests  = [Shapiro]
    for technique in techniques : 
        for test in tests :
            get_value_for_a_test(test, technique,dimensions)
    
    
    