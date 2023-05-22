#in here we will put in the 3 or 4 diffeqns
#maybe we should make them OOP?
#code (as it began) is heavily based of 
#https://www.nature.com/articles/nn.4562#Sec4
#i want to run a simulation model
#each simulation model calls subfunctions for each diffeqn (3 or 4 populations with one diffeqn each)
#each subfunction calls their unique population response functions specific to the population
#the model returns as a dict, params and firing rates for all populations
#%%
from types import SimpleNamespace
import numpy as np
import params
import matplotlib.pyplot as plt

def G_E(x,params=params_dict):
    p=SimpleNamespace(**params)
    if x<p.theta_E:
        return 0
    elif p.theta_E<x<p.theta_E+1/p.m_E:
        return p.m_E*(x-p.theta_E)
    elif x>p.theta_E+1/p.m_E:
        return 1

def G_I_SOM(x,params=params_dict):
    p=SimpleNamespace(**params)
    if x<p.theta_I_SOM:
        return 0
    elif p.theta_I_SOM<x<p.theta_I_SOM+1/p.m_I_SOM:
        return p.m_I_SOM*(x-p.theta_I_SOM)
    elif x>p.theta_I_SOM+1/p.m_I_SOM:
        return 1

def G_I_PV(x,params=params_dict):
    p=SimpleNamespace(**params)
    if x<p.theta_I_PV:
        return 0
    elif p.theta_I_PV<x<p.theta_I_PV+1/p.m_I_PV:
        return p.m_I_PV*(x-p.theta_I_PV)**3
    elif x>p.theta_I_PV+1/p.m_I_PV:
        return 1

def i_E_L23_calculator(surround_size,params=params_dict):
    p=SimpleNamespace(**params)
    if surround_size<1:
        return 0
    else:
        return p.MIN_i_E_L23+(surround_size-1)*p.m_i_E_L23
#%%
def L234_E_PV_SOM_E(params=params_dict):
    p=SimpleNamespace(**params)
    results_dict={}
    ...
    T=np.arange(0,p.T_end,p.dt)
    
    r_E_L23=np.zeros(len(T))
    r_I_PV=np.zeros(len(T))
    r_I_SOM=np.zeros(len(T))
    

    #the following can be modified within a simulation
    i_E_L4=1 # changing the stimulus to the local L4
    surround_size=1 #changing the size of the stimulus and how many of the surrounding horizontal connections agree?

    for i in range(0,len(T)-1):
        dr_E_L23 = p.dt/p.tau_E * (-r_E_L23[i] + G_E(params=params_dict,x=(p.W_EE*r_E_L23[i]-p.W_E_I_PV*r_I_PV[i]-p.W_E_I_SOM*r_I_SOM[i]+p.W_EE_L4*i_E_L4+p.W_EE_L23*i_E_L23_calculator(params=params_dict,surround_size=surround_size))))

        r_E_L23[i+1]=r_E_L23[i]+dr_E_L23

        #

        dr_I_SOM = p.dt/p.tau_I_SOM * (-r_I_SOM[i]+G_I_SOM(params=params_dict,x=(p.W_I_SOM_E*r_E_L23[i]-p.W_I_SOM_I_PV*r_I_PV[i]-p.W_I_SOM_I_SOM*r_I_SOM[i]+p.W_I_SOM_E_L4*i_E_L4+p.W_I_SOM_E_L23*i_E_L23_calculator(params=params_dict,surround_size=surround_size))))
        
        r_I_SOM[i+1]=r_I_SOM[i]+dr_I_SOM

        #

        dr_I_PV= p.dt/p.tau_I_PV * (-r_I_PV[i]+G_I_PV(params=params_dict,x=(p.W_I_PV_E*r_E_L23[i]-p.W_I_PV_I_PV*r_I_PV[i]-p.W_I_PV_I_SOM*r_I_SOM[i]+p.W_I_PV_E_L4*i_E_L4+p.W_I_PV_E_L23*i_E_L23_calculator(params=params_dict,surround_size=surround_size))))

        r_I_PV[i+1]=r_I_PV[i]+dr_I_PV
    
    
    results_dict['params']=params
    results_dict['firing_rates']={'r_E_L23':r_E_L23,'r_I_SOM':r_I_SOM,'r_I_PV':r_I_PV,'Time_series':T}

    return results_dict

#p.dt/p.tau_E(-p.r_E+G_E())
a=L234_E_PV_SOM_E(params=params_dict)
r_E=a['firing_rates']['r_E_L23']
r_I_SOM=a['firing_rates']['r_I_SOM']
r_I_PV=a['firing_rates']['r_I_PV']
Time_series=a['firing_rates']['Time_series']

plt.plot(r_E,Time_series)