#%%
#in here we will put in the 3 or 4 diffeqns
#maybe we should make them OOP?
#code (as it began) is heavily based of 
#https://www.nature.com/articles/nn.4562#Sec4
#i want to run a simulation model
#each simulation model calls subfunctions for each diffeqn (3 or 4 populations with one diffeqn each)
#each subfunction calls their unique population response functions specific to the population
#the model returns as a dict, params and firing rates for all populations

from types import SimpleNamespace
import numpy as np
#from params import params_dict_original as params_dict
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import params as prms
from typing import Callable


def i_E_L23_calculator(surround_size,params:dict):
    p=SimpleNamespace(**params)
    if surround_size<1:
        return 0
    else:
        return p.MIN_i_E_L23+(surround_size-1)*p.m_i_E_L23
#%%
def L234_E_PV_SOM_E(gain_func:Callable,params:dict,input_L4:float,surr_size=1):
    p=SimpleNamespace(**params)
    results_dict={}
    ...
    T=np.arange(0,p.T_end,p.dt)
    
    r_E_L23=np.zeros(len(T))
    r_I_PV=np.zeros(len(T))
    r_I_SOM=np.zeros(len(T))
    
    if gain_func==prms.Gain_sigmoid:
        G_E,G_I_SOM,G_I_PV=prms.Gain_sigmoid,prms.Gain_sigmoid,prms.Gain_sigmoid
    elif gain_func==prms.Gain_ReLU:
        G_E,G_I_SOM,G_I_PV=prms.Gain_ReLU,prms.Gain_ReLU,prms.Gain_ReLU
    elif gain_func==prms.Gain_tanh:
        G_E,G_I_SOM,G_I_PV=prms.Gain_tanh,prms.Gain_tanh,prms.Gain_tanh
    elif gain_func==prms.Gain_exponential:
        G_E=prms.Gain_exponential("E")
        G_I_SOM=prms.Gain_exponential("I_SOM")
        G_I_PV=prms.Gain_exponential("I_PV")

    #the following can be modified within a simulation
    i_E_L4=input_L4 # changing the stimulus to the local L4
    surround_size=surr_size #changing the size of the stimulus and how many of the surrounding horizontal connections agree?

    for i in range(0,len(T)-1):
        dr_E_L23 = p.dt/p.tau_E * (-r_E_L23[i] + G_E(params=params,x=(p.W_EE*r_E_L23[i]-p.W_E_I_PV*r_I_PV[i]-p.W_E_I_SOM*r_I_SOM[i]+p.W_EE_L4*i_E_L4+p.W_EE_L23*i_E_L23_calculator(params=params,surround_size=surround_size))))

        r_E_L23[i+1]=r_E_L23[i]+dr_E_L23

        #

        dr_I_SOM = p.dt/p.tau_I_SOM * (-r_I_SOM[i]+G_I_SOM(params=params,x=(p.W_I_SOM_E*r_E_L23[i]-p.W_I_SOM_I_PV*r_I_PV[i]-p.W_I_SOM_I_SOM*r_I_SOM[i]+p.W_I_SOM_E_L4*i_E_L4+p.W_I_SOM_E_L23*i_E_L23_calculator(params=params,surround_size=surround_size))))
        
        r_I_SOM[i+1]=r_I_SOM[i]+dr_I_SOM

        #

        dr_I_PV= p.dt/p.tau_I_PV * (-r_I_PV[i]+G_I_PV(params=params,x=(p.W_I_PV_E*r_E_L23[i]-p.W_I_PV_I_PV*r_I_PV[i]-p.W_I_PV_I_SOM*r_I_SOM[i]+p.W_I_PV_E_L4*i_E_L4+p.W_I_PV_E_L23*i_E_L23_calculator(params=params,surround_size=surround_size))))

        r_I_PV[i+1]=r_I_PV[i]+dr_I_PV
    
    

    #C_SOM is the contribution of SOM to the total inhibition
    #C_PV is the contribution of PV to the total inhibition
    #dC_SOM is  the contribution of SOM to the change in total inhibition
    #dC_PV is the contribution of PV to the change in total inhibition
    results_dict['params']=params
    results_dict['firing_rates']={'r_E_L23':r_E_L23,'r_I_SOM':r_I_SOM,'r_I_PV':r_I_PV,'Time_series':T,'dC_SOM':(dr_I_SOM/dr_I_SOM+dr_I_PV),'C_SOM':r_I_SOM/r_I_SOM+r_I_PV,'dC_PV':(dr_I_PV/dr_I_SOM+dr_I_PV),'C_PV':r_I_PV/r_I_SOM+r_I_PV,'dC_E':dr_E_L23,'C_E':r_E_L23}

    return results_dict



def FFT_updated(x0,dt:float,T_end:int,freq_range:list=[0,100],plotting=True,title=""): #using the updated scipy functions
    f_S=T_end/dt
    x=x0 - np.nanmean(x0) #nanmean is just the mean of the array
    power=fft(x)
    freqs=fftfreq(len(x)) *f_S
    if plotting:
        plt.figure()
        plt.stem(freqs,np.abs(power))
        plt.xlim(freq_range)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Frequency Domain (Spectrum) Magnitude")
        plt.title(title)
    max_freq=freqs[np.argmax(abs(power))] #most powerful frequency
    max_mag=abs(power[np.argmax(abs(power))]) #power of the most powerful band
    return power,freqs
    




def FFT(x0,plotting,dt,freq_range:list,title=""): 
    #freq_range is used for the xlim eg [0,100] means looking at frequencies between 0 and 100 Hz
    f_s=1000/dt
    x=x0 - np.nanmean(x0) #nanmean is just the mean of the array
    X=fftpack.fft(x) # frequency domain magnitude?
    freqs=fftpack.fftfreq(len(x)) *f_s
    max_freq=freqs[np.argmax(abs(X))] #frequencies corresponding to the indices corresponding to the highest absolute frequencies
    max_mag=abs(X[np.argmax(abs(X))]) #abs of maximum magnitude?
    if plotting:
        plt.figure()
        plt.scatter(freqs,np.abs(X))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Frequency Domain (Spectrum) Magnitude")
        plt.xlim(freq_range) 
        plt.title(title)
    return freqs, X, max_freq, max_mag




