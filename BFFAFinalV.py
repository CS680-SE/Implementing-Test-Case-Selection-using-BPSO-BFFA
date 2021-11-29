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
def BFFA(Eval_Func,n=20,m_i=25,minf=0,dim=None,prog=False,gamma=1.0,beta=0.20,alpha=0.25):
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
