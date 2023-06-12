#run an example simulation
#%%
from eqns import L234_E_PV_SOM_E, FFT_updated
from params import params_dict_original as params_dict_original
from params import params_dict_pfeffer_2013 as params_dict_pfeffer
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, rfft
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure
params_dict_orginal['W_I_SOM_E']=10#making this =10? results in oscillations for the SOM population

params_dict=params_dict_orginal.copy()


params_dict['exp_I_PV']=100
a=L234_E_PV_SOM_E(params=params_dict,surr_size=1)
r_E=a['firing_rates']['r_E_L23']
r_I_SOM=a['firing_rates']['r_I_SOM']

r_I_PV=a['firing_rates']['r_I_PV']
Time_series=a['firing_rates']['Time_series']




x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
power,freqs=FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc",plotting=False)
fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
ax1.plot(Time_series,r_E,label="Exc")
ax1.plot(Time_series,r_I_SOM,label="SOM")
ax1.plot(Time_series,r_I_PV,label="PV")
ax1.set_ylabel("Firing Rate")
ax1.set_xlabel("Time")
ax1.legend(title="Cell Type")

ax2.stem(freqs,np.abs(power))
ax2.set_xlim(0,100)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Power")
ax2.set_title("FFT of Excitatory Population")

#%%
#now I wanna plot the fft transforms

x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc")
#%%
#changing exp_I_PV from 1:5 and plotting the FFT
params_dict=params_dict_orginal.copy()
with PdfPages('../results/FFT_exp_I_PV.pdf') as pages:
    for i in range(0,6):
        
        params_dict['exp_I_PV']=i
        a=L234_E_PV_SOM_E(params=params_dict,surr_size=1)
        
        r_E=a['firing_rates']['r_E_L23']
        r_I_SOM=a['firing_rates']['r_I_SOM']

        r_I_PV=a['firing_rates']['r_I_PV']
        Time_series=a['firing_rates']['Time_series']
        #calculating the FFT
        x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
        power,freqs=FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc",plotting=False)

        
        fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
        ax1.plot(Time_series,r_E,label="Exc")
        ax1.plot(Time_series,r_I_SOM,label="SOM")
        ax1.plot(Time_series,r_I_PV,label="PV")
        ax1.set_ylabel("Firing Rate")
        ax1.set_xlabel("Time")
        ax1.legend(title="Cell Type")
        
        ax2.stem(freqs,np.abs(power))
        ax2.set_xlim(0,100)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")
        ax2.set_title("FFT of Excitatory Population")

        fig1.suptitle("exp_I_PV="+str(i))
        canvas = FigureCanvasPdf(fig1)
        canvas.print_figure(pages)
# %%
#changing surr_size from 1:5 and plotting the FFT
with PdfPages('../results/FFT_surr_size.pdf') as pages:
    for i in range(1,6):
        params_dict=params_dict_orginal.copy()
        a=L234_E_PV_SOM_E(params=params_dict,surr_size=i)
        r_E=a['firing_rates']['r_E_L23']
        r_I_SOM=a['firing_rates']['r_I_SOM']

        r_I_PV=a['firing_rates']['r_I_PV']
        Time_series=a['firing_rates']['Time_series']
        #calculating the FFT
        x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
        power,freqs=FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc",plotting=False)

        fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
        ax1.plot(Time_series,r_E,label="Exc")
        ax1.plot(Time_series,r_I_SOM,label="SOM")
        ax1.plot(Time_series,r_I_PV,label="PV")
        ax1.set_ylabel("Firing Rate")
        ax1.set_xlabel("Time")
        ax1.legend(title="Cell Type")

        ax2.stem(freqs,np.abs(power))
        ax2.set_xlim(0,100)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")
        ax2.set_title("FFT of Excitatory Population")

        fig1.suptitle("surr_size="+str(i))
        canvas = FigureCanvasPdf(fig1)
        canvas.print_figure(pages)
    


# %%
