#run an example simulation
from eqns import L234_E_PV_SOM_E
from eqns import FFT
from params import params_dict
import matplotlib.pyplot as plt
a=L234_E_PV_SOM_E(params=params_dict)
r_E=a['firing_rates']['r_E_L23']
r_I_SOM=a['firing_rates']['r_I_SOM']

r_I_PV=a['firing_rates']['r_I_PV']
Time_series=a['firing_rates']['Time_series']


#%%
plt.plot(Time_series,r_E,label="Exc")
plt.plot(Time_series,r_I_SOM,label="SOM")
plt.plot(Time_series,r_I_PV,label="PV")
plt.ylabel("Firing Rate")
plt.xlabel("Time")
plt.legend(title="Cell Type")


#now I wanna plot the fft transforms

#this is a small stimulus
FFT(x0=a['firing_rates']['r_E_L23'],plotting=True,dt=a['params']['dt'],freq_range=[0,1e2],title="Exc")
#%%
surr_size=2
b=L234_E_PV_SOM_E(params=params_dict,surr_size=surr_size)
FFT(x0=b['firing_rates']['r_E_L23'],plotting=True,dt=b['params']['dt'],freq_range=[0,1e2],title=f"Exc surr_size={surr_size}")

# %%
params_dict['tau_I_PV'],params_dict['tau_I_SOM']=20,20
surr_size=10
b=L234_E_PV_SOM_E(params=params_dict,surr_size=surr_size)
FFT(x0=b['firing_rates']['r_E_L23'],plotting=True,dt=b['params']['dt'],freq_range=[0,1e2],title=f"Exc surr_size={surr_size} tau_E=tau_I")

# %%
