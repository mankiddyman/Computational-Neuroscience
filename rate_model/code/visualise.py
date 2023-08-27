from eqns import L234_E_PV_SOM_E, FFT_updated
from params import params_dict_original as params_dict_original
from params import params_dict_pfeffer_2013 as params_dict_pfeffer
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, rfft
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure

def visualise_sim(sim:dict,title:str=""):
    r_E=sim['firing_rates']['r_E_L23']
    r_I_SOM=sim['firing_rates']['r_I_SOM']

    r_I_PV=sim['firing_rates']['r_I_PV']
    Time_series=sim['firing_rates']['Time_series']
    
    
    
    fig1,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True,dpi=1000)
    
    axs[0].plot(Time_series,r_E,label="PYR",color='blue')
    axs[0].plot(Time_series,r_I_SOM,label="SOM",color='red')
    axs[0].plot(Time_series,r_I_PV,label="PV",color='green')
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$r$(Proportion of local population firing)")
    axs[0].set_ylim(0,1.05)
    #annotate the populations such that they do not overlap by finding the y coordinate corresponding to x=950 
    x_E=np.where(Time_series==950)[0][0]
    y_E=r_E[x_E]
    axs[0].annotate(r"$\mathit{PYR}$",xy=(0.1,0.9),xytext=(900,y_E+.05),fontsize=12,color='blue')
    x_SOM=np.where(Time_series==950)[0][0]
    y_SOM=r_I_SOM[x_SOM]
    axs[0].annotate(r"$\mathit{SOM}$",xy=(0.1,0.9),xytext=(900,y_SOM+.05),fontsize=12,color='red')
    x_PV=np.where(Time_series==950)[0][0]
    y_PV=r_I_PV[x_PV]
    axs[0].annotate(r"$\mathit{PV}$",xy=(0.1,0.9),xytext=(900,y_PV-.05),fontsize=12,color='green')




    #now fft
    #skipping the first 30% of the data
    x0=r_E[int(0.3*len(r_E)):]
    power,freqs=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power=np.abs(power[0:100])
    freqs=freqs[0:100]
    axs[1].plot(freqs,power,label="PYR",color='blue')

    x0=r_I_SOM[int(0.3*len(r_I_SOM)):]
    power,freqs=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power=np.abs(power[0:100])
    freqs=freqs[0:100]
    axs[1].plot(freqs,power,label="SOM",color='red')

    x0=r_I_PV[int(0.3*len(r_I_PV)):]
    power,freqs=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power=np.abs(power[0:100])
    freqs=freqs[0:100]
    axs[1].plot(freqs,power,label="PV",color='green')

    axs[1].set_xlim(0,100)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Power")
    # axs[1,0].stem(freqs_pv,np.abs(power_pv))
    # axs[1,0].set_xlim(0,100)
    # axs[1,0].set_ylim(0,np.max(np.concatenate((np.abs(power_pv),np.abs(power_som)))))
    # axs[1,0].set_xlabel("Frequency (Hz)")
    # axs[1,0].set_ylabel("Power")
    # axs[1,0].set_title("FFT of PV")

    

import eqns
import params as prms
# a=eqns.L234_E_PV_SOM_E(gain_func=prms.Gain_veit,params=default_pars(),input_L4=1,surr_size=0,connectivity_matrix=prms.connectivity['veit'])

# visualise_sim(a)