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
    fig1,axs=plt.subplots(2,2,figsize=(10,5),constrained_layout=True,dpi=1000)
    axs[0,0].plot(Time_series,r_E,label="Exc",color='blue')
    axs[0,0].plot(Time_series,r_I_SOM,label="SOM",color='red')
    axs[0,0].plot(Time_series,r_I_PV,label="PV",color='green')
    axs[0,0].set_ylabel("Firing Rate")
    axs[0,0].set_xlabel("Time")
    #setting legend on the right
    axs[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,title="Cell Type")
    axs[0,0].set_title(title)

    x0=r_E[int(0.3*len(r_E)):]#skipping the first 30% of the data
    power_exc,freqs_exc=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="Exc",plotting=False)
   
    x0=r_I_PV[int(0.3*len(r_I_PV)):]#skipping the first 10% of the data
    power_pv,freqs_pv=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="PV",plotting=False)

    
    
    x0=r_I_SOM[int(0.3*len(r_I_SOM)):]#skipping the first 10% of the data
    power_som,freqs_som=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="SOM",plotting=False)

    max_power_all=max(np.concatenate((np.abs(power_exc),np.abs(power_pv),np.abs(power_som))))

    axs[0,1].stem(freqs_exc,np.abs(power_exc))
    axs[0,1].set_xlim(0,100)
    axs[0,1].set_ylim(0,max_power_all)
    axs[0,1].set_xlabel("Frequency (Hz)")
    axs[0,1].set_ylabel("Power")
    axs[0,1].set_title("FFT of Exc")

    

    axs[1,0].stem(freqs_pv,np.abs(power_pv))
    axs[1,0].set_xlim(0,100)
    axs[1,0].set_ylim(0,np.max(np.concatenate((np.abs(power_pv),np.abs(power_som)))))
    axs[1,0].set_xlabel("Frequency (Hz)")
    axs[1,0].set_ylabel("Power")
    axs[1,0].set_title("FFT of PV")

    

    axs[1,1].stem(freqs_som,np.abs(power_som))
    axs[1,1].set_xlim(0,100)
    axs[1,1].set_ylim(0,np.max(np.concatenate((np.abs(power_pv),np.abs(power_som)))))
    axs[1,1].set_xlabel("Frequency (Hz)")
    axs[1,1].set_ylabel("Power")
    axs[1,1].set_title("FFT of SOM")    
