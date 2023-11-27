import optproblems.cec2005
import numpy as np
import pandas as pd
import os
from platypus import *

# https://platypus.readthedocs.io/en/latest/_modules/platypus/algorithms.html

class interceptedFunction(object):
    """ Normalize returned evaluation types in CEC 2005 functions"""
    def __init__(self, initial_function):
        self.__initFunc = initial_function
    def __call__(self,variables):
        objs = self.__initFunc(variables)
        if isinstance(objs, np.floating):
            objs=[objs]
        return objs
    
    
class bestFitness(Indicator):
    """find best fitness in population"""
    def __init__(self):
        super(bestFitness, self).__init__()
    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation == 0.0]
        if len(feasible) == 0:
            return 0.0
        elif feasible[0].problem.nobjs != 1:
            raise ValueError("bestFitness indicator can only be used for single-objective problems")
        best = None
        optimum = np.min if feasible[0].problem.directions[0] == Problem.MINIMIZE else np.max
        best = optimum([x.objectives[0] for x in feasible])
        return best




 
if __name__ == '__main__':
    # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
    # use single objective functions from optproblems.cec2005
    nexec = 10
    # for all wanted dimensions
    dims = [ 2,10,30,50 ]# 2,10,30,50 are common for all cec functions
    Xovers = [SBX(),SPX()] # TO BE COMPLETED
    Mutations = [PM(),UM()] #all possible Mutations #TO BE COMPLETED
    # build a list of problems for all dimensions
    problems = []
    results = OrderedDict()
    for dim in dims: 
        nfe = 200
        for cec_function in optproblems.cec2005.CEC2005(dim):
            #Platypus problem based on CEC functions using our intercepted class
            problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
            problem.CECProblem = cec_function
            problem.types[:] = Real(-50,50) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
            problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
            # a couple (problem_instance,problem_name) Mandatory because all functions are instance of Problem class
            name = type(cec_function).__name__ + '_' + str(dim) + 'D'
            problems.append((problem, name))
        # a list of (type_algorithm, kwargs_algorithm, name_algorithm)
        algorithms = [(GDE3, dict(variator=DifferentialEvolution()), 'GDE3_' + type(x).__name__ + '_' +
        type(m).__name__) for x in Xovers for m in Mutations]
        results = results | experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,display_stats=True)
    
    print(results)
    
    indicators=[bestFitness()]
    indicators_result = calculate(results, indicators)
    display(indicators_result, ndigits=3)
    
    # create a pandas MultiIndex dataframe
    data=dict()
    for key_algorithm, algorithm in indicators_result.items():
        for key_problem, problem in algorithm.items():
            data[(key_algorithm,key_problem)] = indicators_result[key_algorithm][key_problem]['bestFitness']
    bestFitness_var = pd.DataFrame(data=data)
    print(bestFitness_var)

    #save dataframe to csv file
    os.chdir("./")
    bestFitness_var.to_csv('GDE3_experiment_%d_runs_dim_%s.csv' % (nexec,'_'.join(map(str, dims))))
    # print dataframe statistics
    print(bestFitness_var.describe())
    # reverse MultiIndex levels, bestFitness['Algortihm']['Problem'] -> bestFitness['Problem']['Algortihm']
    bestFitness_var=bestFitness_var.stack(level=0).unstack()
    # plot columns concerned by problem F1_2D
    bestFitness_var['F1_2D'].plot()
    print(bestFitness_var['F1_2D'])
    import matplotlib.pyplot as plt
    plt.show(block=True)

