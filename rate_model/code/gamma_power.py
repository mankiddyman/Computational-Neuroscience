import numpy as np
import pandas as pd
import eqns
import params as prms
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from itertools import chain
import scipy.integrate as integrate
#first pick the gain function, pick the alpha beta desired, run the simulation for multiple surr_size values and plot the following on a 2 y axis scatter plot
#1. gamma power vs surr_size
#2 L2/3 input power vs surr_size

#and another plot
#take an experiment with a given alpha beta and plot the fft for different surr_size values


gain_functions_dict={"gain_veit":{"func":prms.Gain_veit,"alpha":(70),"beta":(60)}}
surr_size_list=np.linspace(0,4,9)

#make a dataframe with columns: gain_function, alpha, beta, surr_size, gamma_power, L2/3 input power

df_gamma_power=pd.DataFrame(columns=["gain_function","alpha","beta","surr_size","fft","gamma_power","L2/3_input_power","sim"],index=range(len(gain_functions_dict.keys())*len(surr_size_list)))


for i,gain_function in enumerate(list(gain_functions_dict.keys())):

    
    for j,surr_size_ in enumerate(surr_size_list):
        print("Doing gain function: "+str(gain_function)+" and surr_size: "+str(surr_size_))

        params_dict=prms.default_pars()
        params_dict["W_EE"]=50
        #refactor this to be a function of halfway between the alpha and beta \emph{range}
        alpha=gain_functions_dict[gain_function]["alpha"]
        beta=gain_functions_dict[gain_function]["beta"]
        func=gain_functions_dict[gain_function]["func"]

        params_dict['W_E_I_PV'],params_dict['W_I_PV_E'],params_dict['W_I_PV_I_PV']=alpha,alpha,alpha
            
        params_dict['W_I_SOM_E'],params_dict['W_E_I_SOM'],params_dict['W_I_PV_I_SOM']=beta,beta/2,beta/3
        
        params_dict['W_I_SOM_I_PV']=alpha*0.01
        params_dict['W_I_SOM_I_SOM']=beta*0.01
        params_dict['W_I_SOM_E_L23']=beta/30
        params_dict['W_I_PV_E_L23']=alpha/100
        params_dict['W_EE_L23']=0         
        #running the sim
        a=eqns.L234_E_PV_SOM_E(gain_func=func,params=params_dict,input_L4=0,surr_size=surr_size_)

        #add sim to ith row of dataframe
        row=i*len(surr_size_list)+j


        df_gamma_power['gain_function'].iloc[row]=gain_function
        df_gamma_power['sim'].iloc[row]=a
        df_gamma_power['alpha'].iloc[row]=alpha
        df_gamma_power['beta'].iloc[row]=beta
        df_gamma_power['surr_size'].iloc[row]=surr_size_


        r_E=a["firing_rates"]["r_E_L23"]
        #skipping the first 30% of the data
        x0=r_E[int(0.3*len(r_E)):]
        power,freqs=eqns.FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="",plotting=False)
        #retain only 0-100Hz
        power=np.abs(power[0:100])
        freqs=freqs[0:100]
        total_power=integrate.trapz(y=power,x=freqs)
        #Gamma power was reported as the peak power at the center frequency of the narrowband peak in the PSD in the 20–30 Hz range.
        gamma_power=integrate.trapz(y=power[np.where((freqs>20) & (freqs<30))],x=freqs[np.where((freqs>20) & (freqs<30))])


        df_gamma_power['fft'].iloc[row]=power
        df_gamma_power['gamma_power'].iloc[row]=gamma_power
        df_gamma_power['L2/3_input_power'].iloc[row]=eqns.i_E_L23_calculator(surround_size=surr_size_,params=params_dict)

#now making the plots

plt.plot(df_gamma_power["surr_size"],df_gamma_power["gamma_power"],label="Gamma power")

for i in range(len(gain_functions_dict.keys())):

    #get data
    df_gamma_power_subset=df_gamma_power[df_gamma_power["gain_function"]==list(gain_functions_dict.keys())[i]]
    plt.plot(df_gamma_power["surr_size"],df_gamma_power["gamma_power"],label="Gamma power")

    fig1,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True,dpi=1000)

    #power vs frequency for different surr_size values
    axs[0].set_title("Gain function: "+str(list(gain_functions_dict.keys())[i]))
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Power")
    axs[0].set_xlim(0,100)
    axs[0].set_ylim(0,0.1)


