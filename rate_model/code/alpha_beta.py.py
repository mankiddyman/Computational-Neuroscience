import numpy as np
#from params import params_dict_aaryan as params_dict
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
import seaborn as sns
from visualise import visualise_sim
import scipy.optimize as opt
import params as prms
import eqns
from scipy.integrate import trapezoid

#analysing alpha beta

#the point of this script is to visualise PV and SOM's contribution to the maximum frequencies/powers for frequencies seperately


#setting up initial parameters

params_dict=prms.default_pars()
#params_dict['tau_E'],params_dict['tau_I_SOM'],params_dict['tau_I_PV']=20,20,20
alpha=1
params_dict['W_E_I_PV'],params_dict['W_I_PV_E'],params_dict['W_I_PV_I_PV']=alpha,alpha,alpha

beta=1
params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E'],params_dict['W_I_PV_I_SOM']=beta,beta,beta/2

#creating a df to store everything
df=pd.DataFrame({"alpha":[],"beta":[],"max_power":[],"mode_freq":[],"sim_data":[]})
alpha_list=[]
beta_list=[]

max_power_exc_list=[]
total_power_exc_list=[]
mode_freq_exc_list=[]
mean_freq_exc_list=[]
n_peaks_exc_list=[]
power_exc_list=[]
freqs_exc_list=[]

max_power_pv_list=[]
total_power_pv_list=[]
mode_freq_pv_list=[]
mean_freq_pv_list=[]
n_peaks_pv_list=[]

power_pv_list=[]
freqs_pv_list=[]

max_power_som_list=[]
total_power_som_list=[]
mode_freq_som_list=[]
mean_freq_som_list=[]
n_peaks_som_list=[]
power_som_list=[]
freqs_som_list=[]

sim_data_list=[]

#we will change alpha and beta from 1 to 10 in steps of 0.1
alpha_range=np.arange(0,100,10)
beta_range=np.arange(0,100,10)
progress=0 #in percent
start=timer()
gain_functions=[prms.Gain_sigmoid,prms.Gain_ReLU,prms.Gain_exponential]
gain_function=prms.Gain_ReLU
for i in list(alpha_range):
    alpha=i
    for j in list(beta_range):
        beta=j
        #updating alpha and beta
        params_dict['W_E_I_PV'],params_dict['W_I_PV_E'],params_dict['W_I_PV_I_PV']=alpha,alpha,alpha
        params_dict['W_E_I_SOM'],params_dict['W_I_SOM_E'],params_dict['W_I_PV_I_SOM']=beta,beta,beta/2
        #running the sim        

        a=eqns.L234_E_PV_SOM_E(gain_func=gain_function,params=params_dict,surr_size=1,input_L4=1)
        progress+=1
        r_E=a['firing_rates']['r_E_L23']
        r_I_SOM=a['firing_rates']['r_I_SOM']
        r_I_PV=a['firing_rates']['r_I_PV']
        Time_series=a['firing_rates']['Time_series']
        
        #adding properties to the lists
        max_power_exc_list.append(a['properties']['Exc']['max_power'])        
        max_power_pv_list.append(a['properties']['PV']['max_power'])
        max_power_som_list.append(a['properties']['SOM']['max_power'])
        
        total_power_exc_list.append(a['properties']['Exc']['total_power'])
        total_power_pv_list.append(a['properties']['PV']['total_power'])
        total_power_som_list.append(a['properties']['SOM']['total_power'])
        
        mode_freq_exc_list.append(a['properties']['Exc']['mode_freq'])
        mode_freq_pv_list.append(a['properties']['PV']['mode_freq'])
        mode_freq_som_list.append(a['properties']['SOM']['mode_freq'])

        mean_freq_exc_list.append(a['properties']['Exc']['mean_freq'])
        mean_freq_pv_list.append(a['properties']['PV']['mean_freq'])
        mean_freq_som_list.append(a['properties']['SOM']['mean_freq'])

        n_peaks_exc_list.append(a['properties']['Exc']['n_peaks'])
        n_peaks_pv_list.append(a['properties']['PV']['n_peaks'])
        n_peaks_som_list.append(a['properties']['SOM']['n_peaks'])

        
        power_exc_list.append(a['properties']['Exc']['power'])
        power_pv_list.append(a['properties']['PV']['power'])
        power_som_list.append(a['properties']['SOM']['power'])

        freqs_exc_list.append(a['properties']['Exc']['freqs'])
        freqs_pv_list.append(a['properties']['PV']['freqs'])
        freqs_som_list.append(a['properties']['SOM']['freqs'])

        sim_data_list.append(a)

        alpha_list.append(alpha)
        beta_list.append(beta)
        end=timer()
        print("\nprogress\n",round(progress/((len(alpha_range))*len(beta_range))*100,2),"%","\ntime_elapsed=\n",timedelta(seconds=end-start),"\nalpha=\n",alpha,"\nbeta=\n",beta,"\n\n")



#now going to visualise max power and mode freq as a function of alpha and beta
df['alpha']=alpha_list
df['beta']=beta_list
df['max_power_exc']=max_power_exc_list
df['max_power_pv']=max_power_pv_list
df['max_power_som']=max_power_som_list
df['total_power_exc']=total_power_exc_list
df['total_power_pv']=total_power_pv_list
df['total_power_som']=total_power_som_list
df['mode_freq_exc']=mode_freq_exc_list
df['mode_freq_pv']=mode_freq_pv_list
df['mode_freq_som']=mode_freq_som_list
df['mean_freq_exc']=mean_freq_exc_list
df['mean_freq_pv']=mean_freq_pv_list
df['mean_freq_som']=mean_freq_som_list
df['n_peaks_exc']=n_peaks_exc_list
df['n_peaks_pv']=n_peaks_pv_list
df['n_peaks_som']=n_peaks_som_list
df['sim_data']=sim_data_list

df['avg_exc']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_E_L23'][int(0.3*len(x['firing_rates']['r_E_L23'])):]))
df['avg_som']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_I_SOM'][int(0.3*len(x['firing_rates']['r_I_SOM'])):]))
df['avg_pv']=df['sim_data'].apply(lambda x: np.mean(x['firing_rates']['r_I_PV'][int(0.3*len(x['firing_rates']['r_I_PV'])):]))
df.to_csv("../results/alpha_beta.tsv",sep="\t",encoding="utf-8")
#df=pd.read_csv("../results/alpha_beta.csv")


#final pivot plot

from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
#%%
fig1,axs =plt.subplots(3,6,figsize=(20,10),dpi=1000,constrained_layout=True)

df.drop_duplicates(['alpha','beta'],inplace=True)


for i,pop in enumerate(["exc","pv","som"]):


    #whether oscillations are present or not
    pivot=df.pivot(index='alpha',columns='beta',values=f'total_power_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,0],annot_kws={"fontsize":8})
    axs[i,0].invert_yaxis()
    axs[i,0].set_title(f"Total_Power_{pop} \n Integral of FFT from 0 to 100 Hz")
    axs[i,0].set_xlabel("Beta (SOM)")
    axs[i,0].set_ylabel("Alpha (PV)")


    pivot=df.pivot(index='alpha',columns='beta',values=f'mean_freq_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,1],annot_kws={"fontsize":8},vmin=0,vmax=100)
    axs[i,1].invert_yaxis()
    axs[i,1].set_title(f"Mean_Freq_{pop}\n Mean of peaks in FFT")
    axs[i,1].set_xlabel("Beta (SOM)")
    axs[i,1].set_ylabel("Alpha (PV)")

    pivot=df.pivot(index='alpha',columns='beta',values=f'n_peaks_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,2],annot_kws={"fontsize":8})
    axs[i,2].invert_yaxis()
    axs[i,2].set_title(f"n_peaks_{pop}\n Number of peaks in FFT")
    axs[i,2].set_xlabel("Beta (SOM)")
    axs[i,2].set_ylabel("Alpha (PV)")


    pivot=df.pivot(index='alpha',columns='beta',values=f'mode_freq_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,3],annot_kws={"fontsize":8},vmin=0,vmax=100)
    axs[i,3].invert_yaxis()
    axs[i,3].set_title(f"Mode_Freq_{pop}")
    axs[i,3].set_xlabel("Beta (SOM)")
    axs[i,3].set_ylabel("Alpha (PV)")

    pivot=df.pivot(index='alpha',columns='beta',values=f'max_power_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,4],annot_kws={"fontsize":8})
    axs[i,4].invert_yaxis()
    axs[i,4].set_title(f"max_power_{pop}")
    axs[i,4].set_xlabel("Beta (SOM)")
    axs[i,4].set_ylabel("Alpha (PV)")


    pivot=df.pivot(index='alpha',columns='beta',values=f'avg_{pop}')
    sns.heatmap(pivot,annot=False,ax=axs[i,5],annot_kws={"fontsize":8},vmin=0,vmax=2)
    axs[i,5].invert_yaxis()
    axs[i,5].set_title(f"avg_{pop}")
    axs[i,5].set_xlabel("Beta (SOM)")
    axs[i,5].set_ylabel("Alpha (PV)")

fig1.suptitle(f"Gain Function: {gain_function.__name__}")
#saving to file
fig1.savefig(f"../results/heatmap_general_{gain_function.__name__}.jpg",dpi=750)
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
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_exc'],color='blue',marker='o',markersize=i,linestyle=':')
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_som'],color='red',marker='o',markersize=i,linestyle=':')
    ax1.plot(df.query(f"beta=={num}")['alpha'],df.query(f"beta=={num}")['avg_pv'],color='green',marker='o',markersize=i,linestyle=':')
ax1.legend(handles=[exc,som,pv,*beta_legend],loc='upper left',bbox_to_anchor=(1,1))

ax2.set_xlabel("Beta (SOM)")
ax2.set_ylabel("Average firing rate")
for i,num in enumerate(df['alpha'].unique()):
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_exc'],color='blue',marker='o',markersize=i,linestyle=':')
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_som'],color='red',marker='o',markersize=i,linestyle=':')
    ax2.plot(df.query(f"alpha=={num}")['beta'],df.query(f"alpha=={num}")['avg_pv'],color='green',marker='o',markersize=i,linestyle=':')
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


alph=80
bet=20
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n Higher PV and low SOM \n (SOM fixation over inhibiting PV and constant EXC)")

alph=10
bet=80
visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n Higher PV and low SOM \n (SOM fixation over inhibiting PV and constant EXC)")

# %%
#heatmap of avg firing rate 

#to do
# make a 9 panel plot of alpha beta and samples
#need to select which samples to use for this


#make a 9 panel heatmap of rate, freq and power across cell population explained by alpha beta

#idenitfy a good measure of the transition to oscillations that is a function of freq, power and rate 

#can then change the activation function and identify effects on oscillation transition

# angle this toward E-I balance by adding up the firing rates of the inhibitory populations and comparing to the excitatory populations

#finish making contribution plot

#examine changes due to different activation function
#relu
#
# %%
