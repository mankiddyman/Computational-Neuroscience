import numpy as np
from params import params_dict_aaryan as params_dict
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
from eqns import *
from timeit import default_timer as timer
from datetime import timedelta
import seaborn as sns
from visualise import visualise_sim
#analysing alpha beta

#the point of this script is to visualise PV and SOM's contribution to the maximum frequencies/powers for frequencies seperately

#we begin by loading params_dict_original

#setting up initial parameters

print(params_dict)
alpha=1
params_dict['W_E_I_PV'],params_dict['W_I_PV_E']=alpha,alpha

beta=1
params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E']=beta,beta

#creating a df to store everything
df=pd.DataFrame({"alpha":[],"beta":[],"max_power":[],"mode_freq":[],"sim_data":[]})
alpha_list=[]
beta_list=[]
max_power_list=[]
mode_freq_list=[]
sim_data_list=[]

#we will change alpha and beta from 1 to 10 in steps of 0.1
alpha_range=np.arange(0,100,10)
beta_range=np.arange(0,100,10)
progress=0 #in percent
start=timer()
for i in list(alpha_range):
    alpha=i
    for j in list(beta_range):
        beta=j
        #updating alpha and beta
        params_dict['W_E_I_PV'],params_dict['W_I_PV_E']=alpha,alpha
        params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E']=beta,beta
        #running the sim        
        a=L234_E_PV_SOM_E(params=params_dict,surr_size=1,input_L4=1)
        progress+=1
        r_E=a['firing_rates']['r_E_L23']
        r_I_SOM=a['firing_rates']['r_I_SOM']
        r_I_PV=a['firing_rates']['r_I_PV']
        Time_series=a['firing_rates']['Time_series']
        x0=r_E[int(0.1*len(r_E)):]#skipping the first 10% of the data
        power,freqs=FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc",plotting=False)
        
        power=np.abs(power)
        #need to save the maximum frequency and power for each alpha and beta
        max_power_list.append(max(power))
        mode_freq_list.append(np.where(power==max(power))[0][0]) #the frequency corresponding to max power
        alpha_list.append(alpha)
        beta_list.append(beta)
        sim_data_list.append(a)
        end=timer()
        print("\nprogress\n",round(progress/((len(alpha_range))*len(beta_range))*100,2),"%","\ntime_elapsed=\n",timedelta(seconds=end-start),"\nalpha=\n",alpha,"\nbeta=\n",beta,"\n\n")



#now going to visualise max power and mode freq as a function of alpha and beta
df['alpha']=alpha_list
df['beta']=beta_list
df['max_power']=max_power_list
df['mode_freq']=mode_freq_list
df['sim_data']=sim_data_list


#wicked
#inspecting 
alph=40
bet=30
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet}")

#final pivot plot
fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),dpi=200,constrained_layout=True)

df.drop_duplicates(['alpha','beta'],inplace=True)

pivot=df.pivot(index='alpha',columns='beta',values='mode_freq')
sns.heatmap(pivot,annot=True,ax=ax1)
ax1.set_title("mode_freq")
pivot=df.pivot(index='alpha',columns='beta',values='max_power')
sns.heatmap(pivot,annot=True,ax=ax2)
ax2.set_title("max_power")