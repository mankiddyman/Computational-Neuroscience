#defining functions to calculate properties of a fft spectrum
#mode_freq
#mean_freq
#n_peaks
#total_power
#max_power
#avg firing rate
from scipy.integrate import trapezoid,cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from visualise import visualise_sim


        
def calculate_properties(sim:dict):
    r_E=sim['firing_rates']['r_E_L23']
    r_I_SOM=sim['firing_rates']['r_I_SOM']
    r_I_PV=sim['firing_rates']['r_I_PV']

    names=['Exc','PV','SOM']
    for i,pop in enumerate([r_E,r_I_PV,r_I_SOM]):
        x0=pop[int(0.3*len(pop)):]#skipping the first 30% of the data

        power,freqs=eqns.FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="",plotting=False)
        power=np.abs(power[0:100]) #restricting to 0-100Hz
        freqs=freqs[0:100]

        total_power=trapezoid(y=power,x=freqs)
        max_power=max(power)
        mode_freq=int(np.where(power==max(power))[0])
        

        indexes=peakutils.indexes(power,thres=0.1,min_dist=10)
        n_peaks=len(indexes)

        mean_freq=np.mean(freqs[indexes])
        avg_firing_rate=np.mean(pop)

        sim['properties'][f"{names[i]}"]={'total_power':total_power,'max_power':max_power,'mode_freq':mode_freq,'n_peaks':n_peaks,'mean_freq':mean_freq,'avg_firing_rate':avg_firing_rate,'power':power,'freqs':freqs}

    return sim

index=28
example=df['sim_data'][index]
a=example
visualise_sim(a)
#visualise_sim(a)
# plt.plot(freqs,power)
# #plt.xlim(0,100)
# plt.title('f(x)')
# plt.show()
power,freqs=eqns.FFT_updated(x0=a['firing_rates']['r_I_SOM'],dt=a['params']['dt'],T_end=a['params']['T_end'],title="",plotting=False)
power=np.abs(power[0:100])
freqs=freqs[0:100]

F_100_0=trapezoid(y=power,x=freqs)



cum_F_X=cumulative_trapezoid(y=power,x=freqs,initial=0)

# plt.plot(freqs,cum_F_X)
# plt.title("F(X)")
# plt.show()


#F'(X)
f_dash_X=np.gradient(power,freqs)

fig,ax=plt.subplots(1,1)

ax.plot(freqs,cum_F_X,label="F(X)")
ax.plot(freqs,f_dash_X,label="f'(X)")
ax.plot(freqs,power,label="f(X)")
#add line for horizontal x axis
ax.axhline(y=0, color='black', linestyle='-')
#detecting peaks
indexes=peakutils.indexes(power,thres=0.1,min_dist=10)

if indexes.size!=0:
    for i in range(0,len(indexes)):
        ax.annotate("peak",(freqs[indexes][i],power[indexes][i]))
ax.legend()
ax.set_title(f"spectra of {index}")
len([i for i in f_dash_X if -2<i<2])