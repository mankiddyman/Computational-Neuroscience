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
    x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
    power,freqs=FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="Exc",plotting=False)
    fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    ax1.plot(Time_series,r_E,label="Exc")
    ax1.plot(Time_series,r_I_SOM,label="SOM")
    ax1.plot(Time_series,r_I_PV,label="PV")
    ax1.set_ylabel("Firing Rate")
    ax1.set_xlabel("Time")
    ax1.legend(title="Cell Type")
    ax1.set_title(title)

    ax2.stem(freqs,np.abs(power))
    ax2.set_xlim(0,100)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_title("FFT of Excitatory Population")
    
