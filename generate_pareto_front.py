from platypus import *
import matplotlib.pyplot as plt
import optproblems
import pandas as pd 

def generation(dimProblem,nbRun):
    #problem,Set the algorithm,Search execution
    problem = DTLZ2(dimProblem)
    
    #SMPSO
    global algorithmSMPSO
    global nondominated_solutionsSMPSO
    algorithmSMPSO = SMPSO(problem, population_size=100)
    algorithmSMPSO.run(nbRun)
    nondominated_solutionsSMPSO = nondominated(algorithmSMPSO.result)
    
    #GDE3
    global algorithmGDE3
    global nondominated_solutionsGDE3
    algorithmGDE3 = GDE3(problem, population_size=100)
    algorithmGDE3.run(nbRun)
    nondominated_solutionsGDE3 = nondominated(algorithmGDE3.result)
    
    '''
    #GeneticAlgorithm
    algorithmGeneticAlgorithm = GeneticAlgorithm(problem, population_size=100)
    algorithmGeneticAlgorithm.run(10000)
    nondominated_solutionsGeneticAlgorithm = nondominated(algorithmGeneticAlgorithm.result)
    '''
def pareto_3D(dimension) :
    #Draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    ax.set_zlabel('f3(x)')
    
    #SMPSO
    xs = [s.objectives[0] for s in nondominated_solutionsSMPSO if s.feasible]
    ys = [s.objectives[1] for s in nondominated_solutionsSMPSO if s.feasible]
    zs = [s.objectives[2] for s in nondominated_solutionsSMPSO if s.feasible]
    ax.scatter(xs, ys, zs,marker="x", c="red")
    
    #GDE3
    xs = [s.objectives[0] for s in nondominated_solutionsGDE3 if s.feasible]
    ys = [s.objectives[1] for s in nondominated_solutionsGDE3 if s.feasible]
    zs = [s.objectives[2] for s in nondominated_solutionsGDE3 if s.feasible]
    ax.scatter(xs, ys, zs,marker="o", c="blue")
    plt.title("Front de Pareto %d Dimensions vers 3 Fonctions d'arrivée (MaO) (DTLZ2)"%dimension)
    
    plt.legend(["SMPSO","GDE3"])

    plt.show()
    
def pareto_2D(dimension):
    #Draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    
    #SMPSO
    xs = [s.objectives[0] for s in nondominated_solutionsSMPSO if s.feasible]
    ys = [s.objectives[1] for s in nondominated_solutionsSMPSO if s.feasible]
    ax.scatter(xs, ys,marker="x", c="red")
    
    #GDE3
    xs = [s.objectives[0] for s in nondominated_solutionsGDE3 if s.feasible]
    ys = [s.objectives[1] for s in nondominated_solutionsGDE3 if s.feasible]
    ax.scatter(xs, ys,marker="o", c="blue")
    
    plt.title("Front de Pareto %d Dimensions vers 2 Fonctions d'arrivée (MO) "%dimension)
    plt.legend(["SMPSO","GDE3"])

    plt.show()


if __name__ == '__main__':
    dimension = 3 
    generation(dimension, 10000)
    pareto_3D(dimension)
    dimension = 2
    generation(dimension, 10000)
    pareto_2D(dimension)