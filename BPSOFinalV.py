

"""Using Libraly"""
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
    def check_dimentions(self,dim):
        None

"""Common Function"""
def random_search(n,dim):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens

"""BPSO"""
def logsig(n): return 1 / (1 + math.exp(-n))
def sign(x): return 1 if x > 0 else (-1 if x!=0 else 0)

def BPSO(Eval_Func,n=20,m_i=200,minf=0,dim=None,prog=False,w1=0.5,c1=1,c2=1,vmax=4):
    """
    input:{ 
            Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            w1: move rate, default=0.5
            c1,c2: It's are two fixed variables, default=1,1
            vmax: Limit search range of vmax, default=4
            }
    output:{
            Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    gens=random_search(n,dim)
    pbest=float("-inf") if minf == 0 else float("inf")
    gbest=float("-inf") if minf == 0 else float("inf")
    #vec=3
    #flag=dr
    gens=random_search(n,dim)
    vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    one_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    zero_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]

    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    pbest=dc(fit)
    xpbest=dc(gens)
    #w1=0.5
    if minf==0:
        gbest=max(fit)
        xgbest=gens[fit.index(max(fit))]
    else:
        gbest=min(fit)
        xgbest=gens[fit.index(min(fit))]

    #c1,c2=1,1
    #vmax=4
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        #w=0.5
        for i in range(n):
            if tuple(gens[i]) in gens_dict:
                score=gens_dict[tuple(gens[i])]
            else:
                score=estimate(gens[i])
                gens_dict[tuple(gens[i])]=score
            fit[i]=score
            if fit[i]>pbest[i] if minf==0 else fit[i]<pbest[i]:#max
                pbest[i]=dc(fit[i])
                xpbest[i]=dc(gens[i])

        if minf==0:
            gg=max(fit)
            xgg=gens[fit.index(max(fit))]
        else:
            gg=min(fit)
            xgg=gens[fit.index(min(fit))]

        if gbest<gg if minf==0 else gbest>gg:#max
            gbest=dc(gg)
            xgbest=dc(xgg)

        oneadd=[[0 for d in range(dim)] for i in range(n)]
        zeroadd=[[0 for d in range(dim)] for i in range(n)]
        c3=c1*random.random()
        dd3=c2*random.random()
        for i in range(n):
            for j in range(dim):
                if xpbest[i][j]==0:
                    oneadd[i][j]=oneadd[i][j]-c3
                    zeroadd[i][j]=zeroadd[i][j]+c3
                else:
                    oneadd[i][j]=oneadd[i][j]+c3
                    zeroadd[i][j]=zeroadd[i][j]-c3

                if xgbest[j]==0:
                    oneadd[i][j]=oneadd[i][j]-dd3
                    zeroadd[i][j]=zeroadd[i][j]+dd3
                else:
                    oneadd[i][j]=oneadd[i][j]+dd3
                    zeroadd[i][j]=zeroadd[i][j]-dd3

        one_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(one_vel,oneadd)]
        zero_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(zero_vel,zeroadd)]
        for i in range(n):
            for j in range(dim):
                if abs(vel[i][j]) > vmax:
                    zero_vel[i][j]=vmax*sign(zero_vel[i][j])
                    one_vel[i][j]=vmax*sign(one_vel[i][j])
        for i in range(n):
            for j in range(dim):
                if gens[i][j]==1:
                    vel[i][j]=zero_vel[i][j]
                else:
                    vel[i][j]=one_vel[i][j]
        veln=[[logsig(s[_s]) for _s in range(len(s))] for s in vel]
        temp=[[random.random() for d in range(dim)] for _n in range(n)]
        for i in range(n):
            for j in range(dim):
                if temp[i][j]<veln[i][j]:
                    gens[i][j]= 0 if gens[i][j] ==1 else 1
                else:
                    pass
    return gbest,xgbest,xgbest.count(1)



"""BFFA"""
def exchange_binary(binary,score):#,alpha,beta,gamma,r):

    #binary in list
    al_binary=binary
    #movement=move(b,alpha,beta,gamma,r)
    movement=math.tanh(score)
    ##al_binary=[case7(b) if random.uniform(0,1) < movement else case8(b) for b in binary]
    if random.uniform(0,1) < movement:
        for i,b in enumerate(binary):
            al_binary[i]=case7(b)
    else:
        for i,b in enumerate(binary):
            al_binary[i]=case8(b)
    return al_binary

def case7(one_bin):
    return 1 if random.uniform(-0.1,0.9)<math.tanh(one_bin) else 0
def case8(one_bin):
    if random.uniform(-0.1,0.9)<math.tanh(int(one_bin)):
        if one_bin==1:
            return 0
        else:return 1
    else:return one_bin
def case9(one_bin,best):
    if random.uniform(0,1)<math.tanh(int(one_bin)):
        return best
    else:return 0

def BFFA(Eval_Func,n=20,m_i=25,minf=0,dim=None,prog=False,gamma=1.0,beta=0.20,alpha=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }
    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    #flag=dr
    global_best=float("-inf") if minf == 0 else float("inf")
    pb=float("-inf") if minf == 0 else float("inf")

    global_position=tuple([0]*dim)
    gen=tuple([0]*dim)
    #gamma=1.0
    #beta=0.20
    #alpha=0.25
    gens_dict = {tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    #gens_dict[global_position]=0.001
    gens=random_search(n,dim)
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




import numpy as np
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm

"""Evaluate Function """
class Evaluate:
    def __init__(self):
        None
    def evaluate(self,gen):
        None
    def check_dimentions(self,dim):
        None

"""Common Function"""
def random_search(n,dim):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens

"""BFFA"""
def exchange_binary(binary,score):#,alpha,beta,gamma,r):

    #binary in list
    al_binary=binary
    #movement=move(b,alpha,beta,gamma,r)
    movement=math.tanh(score)
    ##al_binary=[case7(b) if random.uniform(0,1) < movement else case8(b) for b in binary]
    if random.uniform(0,1) < movement:
        for i,b in enumerate(binary):
            al_binary[i]=case7(b)
    else:
        for i,b in enumerate(binary):
            al_binary[i]=case8(b)
    return al_binary

def case7(one_bin):
    return 1 if random.uniform(-0.1,0.9)<math.tanh(one_bin) else 0
def case8(one_bin):
    if random.uniform(-0.1,0.9)<math.tanh(int(one_bin)):
        if one_bin==1:
            return 0
        else:return 1
    else:return one_bin
def case9(one_bin,best):
    if random.uniform(0,1)<math.tanh(int(one_bin)):
        return best
    else:return 0

def BFFA(Eval_Func,n=20,m_i=25,minf=0,dim=None,prog=False,gamma=1.0,beta=0.20,alpha=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }
    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    #flag=dr
    global_best=float("-inf") if minf == 0 else float("inf")
    pb=float("-inf") if minf == 0 else float("inf")

    global_position=tuple([0]*dim)
    gen=tuple([0]*dim)
    #gamma=1.0
    #beta=0.20
    #alpha=0.25
    gens_dict = {tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    #gens_dict[global_position]=0.001
    gens=random_search(n,dim)
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

""""BBA"""
def BBA(Eval_Func,n=20,m_i=200,dim=None,minf=0,prog=False,qmin=0,qmax=2,loud_A=0.25,r=0.4):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            qmin: frequency minimum to step
            qmax: frequency maximum to step
            loud_A: value of Loudness, default=0.25
            r: Pulse rate, default=0.4, Probability to relocate near the best position
            }
    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    #flag=dr
    #qmin=0
    #qmax=2
    #loud_A=0.25
    #r=0.1
    #n_iter=0
    gens_dic={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    q=[0 for i in range(n)]
    v=[[0 for d in range(dim)] for i in range(n)]
    #cgc=[0 for i in range(max_iter)]
    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    #dr=False
    gens=random_search(n,dim)#[[random.choice([0,1]) for d in range(dim)] for i in range(n)]

    for i in range(n):
        if  tuple(gens[i]) in gens_dic:
            fit[i]=gens_dic[tuple(gens[i])]
        else:
            fit[i]=estimate(gens[i])
            gens_dic[tuple(gens[i])]=fit[i]

    if minf==0:
        maxf=max(fit)
        best_v=maxf
        best_s=gens[fit.index(max(fit))]
    elif minf==1:
        minf=min(fit)
        best_v=minf
        best_s=gens[fit.index(min(fit))]


    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        #cgc[i]=maxf
        for i in range(n):
            for j in range(dim):
                q[i]=qmin+(qmin-qmax)*random.random()
                v[i][j]=v[i][j]+(gens[i][j]-best_s[j])*q[i]

                vstf=abs((2/math.pi)*math.atan((math.pi/2)*v[i][j]))

                if random.random()<vstf:
                    gens[i][j]= 0 if gens[i][j]==1 else 1
                else:
                    pass

                if random.random()>r:
                    gens[i][j]=best_s[j]

            if  tuple(gens[i]) in gens_dic:
                fnew=gens_dic[tuple(gens[i])]
            else:
                fnew=estimate(gens[i])
                gens_dic[tuple(gens[i])]=fnew

            if fnew >= fit[i] and random.random() < loud_A if minf==0 else fnew <= fit[i] and random.random() < loud_A:#max?
                gens[i]=gens[i]
                fit[i]=fnew

            if fnew>best_v if minf==0 else fnew<best_v:
                best_s=dc(gens[i])
                best_v=dc(fnew)

    return best_v,best_s,best_s.count(1)
