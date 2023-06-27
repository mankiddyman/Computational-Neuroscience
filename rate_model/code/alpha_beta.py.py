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


#setting up initial parameters

print(params_dict)
alpha=1
params_dict['W_E_I_PV'],params_dict['W_I_PV_E'],params_dict['W_I_PV_I_PV']=alpha,alpha,alpha

beta=1
params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E'],params_dict['W_I_PV_I_SOM']=beta,beta,beta/2

#creating a df to store everything
df=pd.DataFrame({"alpha":[],"beta":[],"max_power":[],"mode_freq":[],"sim_data":[]})
alpha_list=[]
beta_list=[]
max_power_exc_list=[]
mode_freq_exc_list=[]
max_power_pv_list=[]
mode_freq_pv_list=[]
max_power_som_list=[]
mode_freq_som_list=[]
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
        params_dict['W_E_I_PV'],params_dict['W_I_PV_E'],params_dict['W_I_PV_I_PV']=alpha,alpha,alpha
        params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E'],params_dict['W_I_PV_I_SOM']=beta,beta,beta/2
        #running the sim        
        a=L234_E_PV_SOM_E(params=params_dict,surr_size=1,input_L4=1)
        progress+=1
        r_E=a['firing_rates']['r_E_L23']
        r_I_SOM=a['firing_rates']['r_I_SOM']
        r_I_PV=a['firing_rates']['r_I_PV']
        Time_series=a['firing_rates']['Time_series']
        #calculating fft for each population
        for i,pop in enumerate([r_E,r_I_PV,r_I_SOM]):
            x0=pop[int(0.1*len(pop)):]#skipping the first 10% of the data
            power,freqs=FFT_updated(x0=x0,dt=a['params']['dt'],T_end=a['params']['T_end'],title="Exc",plotting=False)
        
            power=np.abs(power)
            #need to save the maximum frequency and power for each alpha and beta
            if i==0: #r_E
                max_power_exc_list.append(max(power))
                mode_freq_exc_list.append(np.where(power==max(power))[0][0]) #the frequency corresponding to max power
            elif i==1:#r_I_PV
                max_power_pv_list.append(max(power))
                mode_freq_pv_list.append(np.where(power==max(power))[0][0])
            elif i==2: #r_I_SOM
                max_power_som_list.append(max(power))
                mode_freq_som_list.append(np.where(power==max(power))[0][0])
        alpha_list.append(alpha)
        beta_list.append(beta)
        sim_data_list.append(a)
        end=timer()
        print("\nprogress\n",round(progress/((len(alpha_range))*len(beta_range))*100,2),"%","\ntime_elapsed=\n",timedelta(seconds=end-start),"\nalpha=\n",alpha,"\nbeta=\n",beta,"\n\n")



#now going to visualise max power and mode freq as a function of alpha and beta
df['alpha']=alpha_list
df['beta']=beta_list
df['max_power_exc']=max_power_exc_list
df['max_power_pv']=max_power_pv_list
df['max_power_som']=max_power_som_list
df['mode_freq_exc']=mode_freq_exc_list
df['mode_freq_pv']=mode_freq_pv_list
df['mode_freq_som']=mode_freq_som_list
df['sim_data']=sim_data_list
df['avg_EXC']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_E_L23'][int(0.1*len(x['firing_rates']['r_E_L23'])):]))
df['avg_SOM']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_I_SOM'][int(0.1*len(x['firing_rates']['r_I_SOM'])):]))
df['avg_PV']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_I_PV'][int(0.1*len(x['firing_rates']['r_I_PV'])):]))
df.to_csv("../results/alpha_beta.tsv",sep="\t",encoding="utf-8")
#df=pd.read_csv("../results/alpha_beta.csv")

#wicked
#inspecting 
alph=40
bet=30
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet}")

#final pivot plot
#%%
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

fig1,axs =plt.subplots(3,2,figsize=(10,10),dpi=200,constrained_layout=True)

df.drop_duplicates(['alpha','beta'],inplace=True)


for i,pop in enumerate(["exc","pv","som"]):

    pivot=df.pivot(index='alpha',columns='beta',values=f'mode_freq_{pop}')
    sns.heatmap(pivot,annot=True,ax=axs[i,0])
    axs[i,0].invert_yaxis()
    axs[i,0].set_title(f"Mode_Freq_{pop}")
    axs[i,0].set_xlabel("Beta (SOM)")
    axs[i,0].set_ylabel("Alpha (PV)")

    pivot=df.pivot(index='alpha',columns='beta',values=f'max_power_{pop}')
    sns.heatmap(pivot,annot=True,ax=axs[i,1])
    axs[i,1].invert_yaxis()
    axs[i,1].set_title(f"max_power_{pop}")
    axs[i,1].set_xlabel("Beta (SOM)")
    axs[i,1].set_ylabel("Alpha (PV)")

#%%
#visualise no PV and no SOM
alph=0
bet=0
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n No PV and No SOM")


alph=90
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

#plot power against alpha beta ig

#plot power against freq coloured by alpha beta

#plot of Average firing rate for each population with alpha on the x axis, firing rates on the y axis and different marker sizes for different beta values



#%%
fig2,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),dpi=500,constrained_layout=True)

#setting up legend
import matplotlib.lines as mlines
exc=mlines.Line2D([], [], color='blue', marker='o', linestyle='None',label="EXC")
som=mlines.Line2D([], [], color='red', marker='o', linestyle='None',label="SOM")
pv=mlines.Line2D([], [], color='green', marker='o', linestyle='None',label="PV")

alpha_legend=[mlines.Line2D([],[],color='black',marker='o',linestyle='None',markersize=i+0.5,label=f"alpha={num}") for i,num in enumerate(df['alpha'].unique())]
beta_legend=[mlines.Line2D([],[],color='black',marker='o',linestyle='None',markersize=i+0.5,label=f"beta={num}") for i,num in enumerate(df['beta'].unique())]
#plotting
ax1.set_xlabel("Alpha (PV)")
ax1.set_ylabel("Average firing rate")
for i,num in enumerate(df['beta'].unique()):
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_EXC'],color='blue',marker='o',markersize=i,linestyle=':')
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_SOM'],color='red',marker='o',markersize=i,linestyle=':')
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_PV'],color='green',marker='o',markersize=i,linestyle=':')
ax1.legend(handles=[exc,som,pv,*beta_legend],loc='upper left',bbox_to_anchor=(1,1))

ax2.set_xlabel("Beta (SOM)")
ax2.set_ylabel("Average firing rate")
for i,num in enumerate(df['alpha'].unique()):
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_EXC'],color='blue',marker='o',markersize=i,linestyle=':')
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_SOM'],color='red',marker='o',markersize=i,linestyle=':')
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_PV'],color='green',marker='o',markersize=i,linestyle=':')
ax2.legend(handles=[exc,som,pv,*alpha_legend],loc='upper left',bbox_to_anchor=(1,1))


# %%
alph=70
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

# %%
alph=60
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

alph=50
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

alph=40
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")


# %%
