"""Using Library"""
import numpy as np
import csv
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm

 
# opening the CSV file
with open('paintcontrol.csv', mode ='r')as file:
  # reading the CSV file
  csvFile = csv.reader(file)
  # displaying the contents of the CSV file
  for lines in csvFile:
        print(lines)
        
"""Evaluate Function """
class Evaluate:
    def __init__(self):
        None
    def evaluate(self,gen):
        None
    def check_dimentions(self,D):
        None

"""Common Function"""
def random_search(n,D):
    """
    create genes list
    input:{ n: Number of population, default=20
            D: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(D)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,D)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens
def BFFA(Eval_Func,n=20,m_i=25,minf=0,D=None,prog=False,gamma=1.0,beta=0.20,alpha=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            D: Number of dimension , default=None
            prog: Do you want to use a progress bar?, default=False
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Number of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if D==None:
       D=Eval_Func().check_dimentions(D)
    #flag=dr
    global_best=float("-inf") if minf == 0 else float("inf")
    pb=float("-inf") if minf == 0 else float("inf")
    global_position=tuple([0]*D)
    gen=tuple([0]*D)
    #gamma=1.0
    #beta=0.20
    #alpha=0.25
    gens_dict = {tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}

    #gens_dict[global_position]=0.001
    gens=random_search(n,D)
    #vs = [[random.choice([0,1]) for i in range(length)] for i in range(N)]
    for gen in gens:
        if tuple(gen) in gens_dict:
            score = gens_dict[tuple(gen)]
        else:
            score=estimate(gen)
            gens_dict[tuple(gen)]=score
        if score > global_best:
            global_best=score
            global_position=dc(gen)
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        for i,x in enumerate(gens):
            for j,y in enumerate(gens):
                if gens_dict[tuple(y)] < gens_dict[tuple(x)]:
                    gens[j]=exchange_binary(y,gens_dict[tuple(y)])
                gen = gens[j]
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=estimate(gens[j])
                    gens_dict[tuple(gen)]=score
                if score > global_best if minf==0 else score < global_best:
                    global_best=score
                    global_position=dc(gen)
    return global_best,global_position,global_position.count(1)
