import matplotlib.pyplot as plt
import numpy as np
from typing import Callable#so I can type hint a function
from params import *
import seaborn as sns

sns.set_theme(context='paper',style='ticks')
#first figure for dissertation
#the gain function diagram for veit et al 2017, with colours for each cell type with an xlim corresponding to the range of the inputs which is... -20,20


#%%
def plot_Gain_function(xlim_list:list,title:str,pars:dict,annotation_height:float=0.9,ylim_list=[0,1.1],**kwargs):
    """
    Plots the gain function of a neuron with range [0,1]

    supply functions as kwargs"""
    x=np.linspace(-500,500,1000) #figure out a good boundary
    #E, PV, SOM
    fig=plt.figure(dpi=300)
    colors=["blue","green","red"]
    for i,func in enumerate(kwargs.values()):   
        print("doing function",func.__name__)  
        y=[func(x_coord,params=pars) for x_coord in x]
        #find the index of the first value of y that is greater than 0.9
        annotation=np.where(np.array(y)>annotation_height)[0][0]
        x[annotation]
        plt.annotate(rf"${list(kwargs.keys())[i]}$",xy=(x[annotation]-2,0.9),xytext=(x[annotation]-3.25,annotation_height),fontsize=12,color=colors[i])
        label=list(kwargs.keys())[i]
        plt.plot(x,y,label=label,color=colors[i],linewidth=1.5,linestyle='-')
    
    plt.plot(x,x,label="y=x",color='orange')
    plt.annotate(r"$y=x$",xy=(-2,0.8),xytext=(-2.5,0.9),fontsize=12,color="orange")
    plt.ylim(ylim_list[0],ylim_list[1])
    plt.xlim(xlim_list[0],xlim_list[1])
    plt.xlabel(r"$x$ (Weighted sum of synaptic inputs)",fontsize=12)
    plt.ylabel(r"$G(x)$",fontsize=12)
    #put legend outside of plot
    #plt.legend(bbox_to_anchor=(.8, 1), loc='upper left', borderaxespad=0.)
    plt.title(title)
#    plt.show()
    return fig
#%%

def visualise_sim(sim:dict,title:str=""):
    r_E=sim['firing_rates']['r_E_L23']
    r_I_SOM=sim['firing_rates']['r_I_SOM']

    r_I_PV=sim['firing_rates']['r_I_PV']
    Time_series=sim['firing_rates']['Time_series']
    
    
    
    fig1=plt.figure(dpi=300)
    
    plt.plot(Time_series,r_E,label="PYR",color='blue')
    plt.plot(Time_series,r_I_SOM,label="SOM",color='red')
    plt.plot(Time_series,r_I_PV,label="PV",color='green')
    plt.xlabel("Time",fontsize=12)
    plt.ylabel(r"$r$ (Proportion of population firing)",fontsize=10)
    plt.ylim(0,1.05)
    #annotate the populations such that they do not overlap by finding the y coordinate corresponding to x=950 
    x_E=np.where(Time_series==950)[0][0]
    y_E=r_E[x_E]
    plt.annotate(r"$\mathit{PYR}$",xy=(0.1,0.9),xytext=(900,y_E+.05),fontsize=12,color='blue')
    x_SOM=np.where(Time_series==950)[0][0]
    y_SOM=r_I_SOM[x_SOM]
    plt.annotate(r"$\mathit{SOM}$",xy=(0.1,0.9),xytext=(900,y_SOM+.05),fontsize=12,color='red')
    x_PV=np.where(Time_series==950)[0][0]
    y_PV=r_I_PV[x_PV]
    plt.annotate(r"$\mathit{PV}$",xy=(0.1,0.9),xytext=(900,y_PV-.1),fontsize=12,color='green')
    plt.title(r'Time series of $r$',fontsize=12)
    plt.suptitle(r'from $\mathit{Veit\ et\ al.\ 2017}$',fontsize=8)

    fig2=plt.figure(dpi=300)


    #now fft
    #skipping the first 30% of the data
    x0=r_E[int(0.3*len(r_E)):]
    power_E,freqs_E=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power_E=np.abs(power_E[0:100])
    freqs_E=freqs_E[0:100]
    plt.plot(freqs_E,power_E,label="PYR",color='blue',linestyle='-')

    x0=r_I_SOM[int(0.3*len(r_I_SOM)):]
    power_SOM,freqs_SOM=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power_SOM=np.abs(power_SOM[0:100])
    freqs_SOM=freqs_SOM[0:100]
    plt.plot(freqs_SOM,power_SOM,label="SOM",color='red',linestyle='--')

    x0=r_I_PV[int(0.3*len(r_I_PV)):]
    power_PV,freqs_PV=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power_PV=np.abs(power_PV[0:100])
    freqs_PV=freqs_PV[0:100]
    plt.plot(freqs_PV,power_PV,label="PV",color='green',linestyle='-.')
    plt.title(r'Power spectra of $r$',fontsize=12)
    plt.suptitle(r'from $\mathit{Veit\ et\ al.\ 2017}$',fontsize=8)
#now add annotations at the very top right by getting max y coordinate of all 3 powers
    y_annotation=np.max(np.concatenate((power_E,power_SOM,power_PV)))
    
    plt.annotate(r"$\mathit{PYR}$",xy=(0.1,0.9),xytext=(90,y_annotation),fontsize=12,color='blue')
    plt.annotate(r"$\mathit{PV}$",xy=(0.1,0.9),xytext=(90,0.9*y_annotation),fontsize=12,color='green')
    plt.annotate(r"$\mathit{SOM}$",xy=(0.1,0.9),xytext=(90,0.8*y_annotation),fontsize=12,color='red')
    

    plt.xlim(0,100)
    plt.xlabel(r"Frequency $(Hz)$",fontsize=12)
    plt.ylabel("Power (AU)",fontsize=12)
    # axs[1,0].stem(freqs_pv,np.abs(power_pv))
    # axs[1,0].set_xlim(0,100)
    # axs[1,0].set_ylim(0,np.max(np.concatenate((np.abs(power_pv),np.abs(power_som)))))
    # axs[1,0].set_xlabel("Frequency (Hz)")
    # axs[1,0].set_ylabel("Power")
    # axs[1,0].set_title("FFT of PV")
    return fig1,fig2
    


fig1=plot_Gain_function(xlim_list=[-20,20],title=r' Gain Functions from $\mathit{Veit\ et\ al.\ 2017}$',pars=params,PYR=Gain_veit_E,SOM=Gain_veit_I_SOM,PV=Gain_veit_I_PV)
#save this
fig1.savefig("../results/figures/fig_1_gain_function_veit.png")

# plot_Gain_function(pars=params,exp_E=Gain_exponential_E,Gain_exponential_I_PV=Gain_tanh_I_PV,Gain_exponential_I_SOM=Gain_tanh_I_SOM)

# plot_Gain_function(pars=params,exp_I_SOM=Gain_exponential_I_SOM,tanh_I_SOM=Gain_tanh_I_SOM)


#next figure has two panels 1st is the rates time series for each population in the default veit model. 2nd is the power freqs plot

#first generte a simulation
import eqns
import params as prms
import visualise
a=eqns.L234_E_PV_SOM_E(gain_func=prms.Gain_veit,params=prms.default_pars(),input_L4=1,surr_size=0,connectivity_matrix=prms.connectivity['veit'])

fig2,fig3=visualise_sim(a)
fig2.savefig("../results/figures/fig_2_time_series_veit.png")
fig3.savefig("../results/figures/fig_3_power_spectra_veit.png")

#now I want to plot the gamma power vs surround size for the veit model
#first I need to generate a dataframe with the gamma power for each surround size
#%%
import pandas as pd
import scipy.integrate as integrate
import warnings
n_rows=51
df=pd.DataFrame(columns=['sim_data','surr_size','power_array','freqs_array','gamma_power','total_power','gamma_proportion'],index=np.linspace(0,100,n_rows))
#now I want to loop through the surround sizes and generate a simulation for each one
for i in np.linspace(0,50,n_rows):
    sim=None
    del sim
    sim=eqns.L234_E_PV_SOM_E(gain_func=prms.Gain_veit,params=prms.default_pars(),input_L4=1,surr_size=i,connectivity_matrix=prms.connectivity['veit']).copy()
    df['sim_data'].iloc[int(i)]=[sim.copy()]
    df['surr_size'].iloc[int(i)]=i
    #now I want to calculate the gamma power
    x0=sim['firing_rates']['r_E_L23'][int(0.3*len(sim['firing_rates']['r_E_L23'])):]
    power_E,freqs_E=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="",plotting=False)
    #retain only 0-100Hz
    power_E=np.abs(power_E[0:100])
    freqs_E=freqs_E[0:100]
    total_power=integrate.trapz(y=power_E,x=freqs_E)
    #Gamma power was reported as the peak power at the center frequency of the narrowband peak in the PSD in the 20–30 Hz range.
    gamma_power=integrate.trapz(y=power_E[np.where((freqs_E>20) & (freqs_E<30))],x=freqs_E[np.where((freqs_E>20) & (freqs_E<30))])
    df['power_array'].iloc[int(i)]=power_E
    df['freqs_array'].iloc[int(i)]=freqs_E
    df['gamma_power'].iloc[int(i)]=gamma_power
    df['total_power'].iloc[int(i)]=total_power
    df['gamma_proportion'].iloc[int(i)]=gamma_power/total_power
    
fig4=plt.figure(dpi=300)
plt.plot(df['surr_size'],df['gamma_power'],linewidth=1.5,linestyle='-',color='blue')
plt.xlabel("Surround size (AU)",fontsize=12)
plt.ylabel(" Gamma power (AU)",fontsize=12)
plt.title(r'from $\mathit{Veit\ et\ al.\ 2017}$',fontsize=12)
fig4.savefig("../results/figures/fig_4_gamma_power_veit.png")


#now figure 5 is the alpha beta heatmap for the veit model
#%%
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
from itertools import chain
import copy #for making deepcopies of dict

#analysing alpha beta

#the point of this script is to visualise PV and SOM's contribution to the maximum frequencies/powers for frequencies seperately


#setting up initial parameters
connectivity_original=copy.deepcopy(prms.connectivity)
connectivity=connectivity_original
#we will change alpha and beta from 1 to 10 in steps of 0.1
start_ab=0
stop_ab=30
alpha_range=np.around(np.linspace(start=start_ab,stop=stop_ab,num=11),2)
beta_range=np.around(np.linspace(start=start_ab,stop=stop_ab,num=11),2)
progress=0 #in percent
start=timer()
# gain_functions=[prms.Gain_veit,prms.Gain_exponential,prms.Gain_tanh]
gain_functions=[prms.Gain_veit]
connectivity_matrices=list(connectivity.keys())[1:2]

for _function in gain_functions:
    for connectivity_matrix_name in connectivity_matrices:
        gain_function=_function
        print("Doing gain function: ",_function.__name__,"with connectivity",connectivity_matrix_name)
        
        params_dict=prms.default_pars()
        

        
        
        

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
        connectivity_matrix_list=[]



        for i in list(alpha_range):
            alpha=i
            for j in list(beta_range):
                beta=j
                #updating alpha and beta
                scale=max(alpha_range)/2
                #W_EE
                connectivity=copy.deepcopy(prms.connectivity)
                
                connectivity[connectivity_matrix_name][0,0]*=1
                #W_I_PV_E
                connectivity[connectivity_matrix_name][0,1]*=alpha/scale
                #W_I_SOM_E
                connectivity[connectivity_matrix_name][0,2]*=beta/scale
                #W_E_I_PV
                connectivity[connectivity_matrix_name][1,0]*=alpha/scale
                #W_I_PV_I_PV
                connectivity[connectivity_matrix_name][1,1]*=alpha/ scale
                #W_I_SOM_I_PV
                connectivity[connectivity_matrix_name][1,2]*=0/scale
                #W_E_I_SOM
                connectivity[connectivity_matrix_name][2,0]*=beta/scale
                #W_I_PV_I_SOM
                connectivity[connectivity_matrix_name][2,1]*=beta/scale
                #W_I_SOM_I_SOM
                connectivity[connectivity_matrix_name][2,2]*=0/scale
                #W_E_I_L23
                connectivity[connectivity_matrix_name][3,0]*=0/scale
                #W_I_PV_E_L23
                connectivity[connectivity_matrix_name][3,1]*=alpha/scale
                #W_I_SOM_E_L23
                connectivity[connectivity_matrix_name][3,2]*=beta/scale
                #W_E_I_L4
                connectivity[connectivity_matrix_name][4,0]*=scale/scale
                #W_I_PV_E_L4
                connectivity[connectivity_matrix_name][4,1]*=alpha/scale
                #W_I_SOM_E_L4
                connectivity[connectivity_matrix_name][4,2]*=beta/scale

                #running the sim        

                a=eqns.L234_E_PV_SOM_E(gain_func=gain_function,params=params_dict,surr_size=1,input_L4=1,connectivity_matrix=copy.deepcopy(connectivity[connectivity_matrix_name]))
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
                connectivity_matrix_list.append([connectivity_matrix_name,connectivity[connectivity_matrix_name]])
                end=timer()
                print("\nprogress\n",round(progress/((len(alpha_range))*len(beta_range)*len(gain_functions)*len(connectivity_matrices))*100,2),"%","\ntime_elapsed=\n",timedelta(seconds=end-start),"\nalpha=\n",alpha,"\nbeta=\n",beta,"\n\n")



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

        #adding Gain function inputs to the dataframe
        df['gain_input_exc']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_exc'])
        df['gain_input_pv']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_pv'])
        df['gain_input_som']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_som'])
        df.to_csv(f"../results/alpha_beta_{min(alpha_range)}_{max(alpha_range)}_{_function.__name__}.tsv",sep="\t",encoding="utf-8")
        #df=pd.read_csv("../results/alpha_beta.csv")
        #visualising the range of gain function inputs:
        aggregate_gain_inputs=[]
        aggregate_input_E=[]
        aggregate_input_PV=[]
        aggregate_input_SOM=[]
        for pop in ['exc','pv','som']:

            col=df[f'gain_input_{pop}']
            for i in range(len(df)):
            
                aggregate_gain_inputs.append(df.iloc[i][f'gain_input_{pop}'])
                if pop=='exc':
                    aggregate_input_E.append(df.iloc[i][f'gain_input_{pop}'])
                elif pop=='pv':
                    aggregate_input_PV.append(df.iloc[i][f'gain_input_{pop}'])
                elif pop=='som':
                    aggregate_input_SOM.append(df.iloc[i][f'gain_input_{pop}'])

        aggregate_input_E=list(chain.from_iterable(aggregate_input_E))
        aggregate_input_PV=list(chain.from_iterable(aggregate_input_PV))
        aggregate_input_SOM=list(chain.from_iterable(aggregate_input_SOM))
        aggregate_gain_inputs=list(chain.from_iterable(aggregate_gain_inputs))


        #final pivot plot

        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
        #%%
        fig5,axs =plt.subplots(3,4,figsize=(12,12*3/4),dpi=500,constrained_layout=True)

        df.drop_duplicates(['alpha','beta'],inplace=True)


        for i,pop in enumerate(["exc","pv","som"]):


            #whether oscillations are present or not
            pivot=df.pivot(index='alpha',columns='beta',values=f'total_power_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,0],annot_kws={"fontsize":8})
            axs[i,0].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,0].set_title(f"Total Power {pop.upper()} \n"+ r"Integral of FFT from $0$ to $100$ $Hz$")
            if pop=='PYR':
                pop='exc'
            axs[i,0].set_xlabel(r"$\beta$ (SOM)")
            axs[i,0].set_ylabel(r"$\alpha$ (PV)")

            
            pivot=df.pivot(index='alpha',columns='beta',values=f'avg_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,1],annot_kws={"fontsize":8},vmin=0,vmax=1.5)
            axs[i,1].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,1].set_title(f"{pop.upper()}\n"+r" Average $r$")
            if pop=='PYR':
                pop='exc'
            axs[i,1].set_xlabel(r"$\beta$ (SOM)")
            axs[i,1].set_ylabel(r"$\alpha$ (PV)")

            pivot=df.pivot(index='alpha',columns='beta',values=f'gain_input_{pop}')
            pivot=pivot.applymap(np.mean)
            sns.heatmap(pivot,annot=False,ax=axs[i,2],annot_kws={"fontsize":8},vmin=min(aggregate_gain_inputs),vmax=max(aggregate_gain_inputs))
            axs[i,2].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,2].set_title(f"{pop.upper()}\n Mean Synaptic Input")
            if pop=='PYR':
                pop='exc'
            axs[i,2].set_xlabel(r"$\beta$ (SOM)")
            axs[i,2].set_ylabel(r"$\alpha$ (PV)")

            
            
                
            pivot=df.pivot(index='alpha',columns='beta',values=f'n_peaks_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,3],annot_kws={"fontsize":8})
            axs[i,3].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,3].set_title(f"{pop.upper()}\n Number of peaks in FFT")
            if pop=='PYR':
                pop='exc'
            axs[i,3].set_xlabel(r"$\beta$ (SOM)")
            axs[i,3].set_ylabel(r"$\alpha$ (PV)")

            
        fig5.suptitle(r"Gain Function and  Connectivity_matrix derived from $\mathit{Veit\ et\ al.\ 2017}$",fontsize=12)
        #saving to file
        fig5.savefig(f"../results/figures/fig_5_alpha_beta_{min(alpha_range)}_{max(alpha_range)}_heatmap_general_{gain_function.__name__}_{connectivity_matrix_name}.jpg",format="jpg",dpi=750)
        plt.show()
        plt.close()
        #%%
        #visualise no PV and no SOM
        # alph=0
        # bet=0
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n No PV and No SOM")


        # alph=90
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

        # #plot power against alpha beta ig

        # #plot power against freq coloured by alpha beta

        # #plot of Average firing rate for each population with alpha on the x axis, firing rates on the y axis and different marker sizes for different beta values

#%%
#now We will show the Dlx models but i will finish this when i get home #I DID MOVE THIS FORWARD WHEN I GOT HOME
#it will be a version of rates/time series with different betas 
#then power spectra with different betas
#then gamma power vs surround size with different betas

#first generte the required simulations
#based upon the alpha beta plot, the required coordinates lie in the range
 

#now I will generate a simulation for each coordinate

import numpy as np
import pandas as pd
import params as prms
import copy
import eqns
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
#storing the simulations in a dataframe
coords={"beta":np.array([24]),"alpha":np.linspace(16,21,5)}

#forming coords for the timeseries and power spectra plots
start_ab=0
stop_ab=30
range_ab=np.linspace(start=start_ab,stop=stop_ab,num=11)
scale_of_heatmap=max(range_ab)/2
#next forming a list of surr sizes for each coordinate
surr_sizes=np.linspace(0,50,51)
surr_sizes=np.repeat(surr_sizes,len(coords['alpha'])*len(coords['beta']))
nrow=len(surr_sizes)

coords['beta']=np.tile(coords['beta'],int(nrow/len(coords['beta'])))
coords['alpha']=np.tile(coords['alpha'],int(nrow/len(coords['alpha'])))
if 'df' in locals():
    del df

df=pd.DataFrame(columns=['alpha','beta','surr_size','sim_data'],index=range(0,nrow))


for i in range(0,len(df)):
    
    alpha=coords['alpha'][i]
    beta=coords['beta'][i]
    surr_size=surr_sizes[i]

    df['surr_size'].iloc[i]=surr_sizes[i]
    df['alpha'].iloc[i]=alpha
    df['beta'].iloc[i]=beta
    
    connectivity=copy.deepcopy(prms.connectivity['veit'])
    #updating alpha and beta
    scale=scale_of_heatmap
    connectivity=copy.deepcopy(prms.connectivity['veit'])
    #W_EE
    connectivity[0,0]*=1
    #W_I_PV_E
    connectivity[0,1]*=alpha/scale
    #W_I_SOM_E
    connectivity[0,2]*=beta/scale
    #W_E_I_PV
    connectivity[1,0]*=alpha/scale
    #W_I_PV_I_PV
    connectivity[1,1]*=alpha/ scale
    #W_I_SOM_I_PV
    connectivity[1,2]*=0/scale
    #W_E_I_SOM
    connectivity[2,0]*=beta/scale
    #W_I_PV_I_SOM
    connectivity[2,1]*=beta/scale
    #W_I_SOM_I_SOM
    connectivity[2,2]*=0/scale
    #W_E_I_L23
    connectivity[3,0]*=0/scale
    #W_I_PV_E_L23
    connectivity[3,1]*=alpha/scale
    #W_I_SOM_E_L23
    connectivity[3,2]*=beta/scale
    #W_E_I_L4
    connectivity[4,0]*=scale/scale
    #W_I_PV_E_L4
    connectivity[4,1]*=alpha/scale
    #W_I_SOM_E_L4
    connectivity[4,2]*=beta/scale
    
    sim=None
    del sim

    sim=eqns.L234_E_PV_SOM_E(gain_func=prms.Gain_veit,params=prms.default_pars(),input_L4=1,surr_size=surr_size,connectivity_matrix=connectivity).copy()
    df['sim_data'].iloc[i]=[sim.copy()]
    print(i)

#test if the simulations are correct
for i in range(0,len(df)):
    #print(df['sim_data'].iloc[i][0]['firing_rates']['r_E_L23'])
    print(np.mean(df['sim_data'].iloc[i][0]['firing_rates']['r_E_L23']))
#should change




#note to self when I wake up I got as far as generating the simulations, but i didnt finish testing them, next i have to add the collumns to generate the time series and power spectra plots and the gamma power vs surround size plots and then i have to generate the plots :'|

#ok testing has completed, now I will add the collumns 

#first the time series
df['r_E']=df['sim_data'].apply(lambda x: x[0]['firing_rates']['r_E_L23'])
df['r_I_PV']=df['sim_data'].apply(lambda x: x[0]['firing_rates']['r_I_PV'])
df['r_I_SOM']=df['sim_data'].apply(lambda x: x[0]['firing_rates']['r_I_SOM'])

#now the power spectra
df['power_E']=df['sim_data'].apply(lambda x: x[0]['properties']['Exc']['power'])
df['power_PV']=df['sim_data'].apply(lambda x: x[0]['properties']['PV']['power'])
df['power_SOM']=df['sim_data'].apply(lambda x: x[0]['properties']['SOM']['power'])
#and freqs (they happen to be identical and defined by np.linspace(0,100,1000) or something like that)
df['freqs_E']=df['sim_data'].apply(lambda x: x[0]['properties']['Exc']['freqs'])
df['freqs_PV']=df['sim_data'].apply(lambda x: x[0]['properties']['PV']['freqs'])
df['freqs_SOM']=df['sim_data'].apply(lambda x: x[0]['properties']['SOM']['freqs'])

#now the gamma power vs surround size
gamma_power_list=[]
for i in range(0,len(df)):
    gamma_power_list.append(integrate.trapz(y=df['power_E'].iloc[i][np.where((df['freqs_E'].iloc[i]>20) & (df['freqs_E'].iloc[i]<30))],x=df['freqs_E'].iloc[i][np.where((df['freqs_E'].iloc[i]>20) & (df['freqs_E'].iloc[i]<30))]))
df['gamma_power']=gamma_power_list


#ok now ready to plot
#first the time series

fig6=plt.figure(figsize=(10,10*3/4),dpi=500)
#want to show 3 beta values
alphas_=np.unique(np.linspace(coords['alpha'].min(),coords['alpha'].max(),3))
betas_=np.unique(np.linspace(coords['beta'].min(),coords['beta'].max(),3))
for i in range(0,len(alphas_)):
    for j in range(0,len(betas_)):
        alpha=alphas_[i]
        beta=betas_[j]
        df2=df.query(f"alpha=={alpha} & beta=={beta}& surr_size==0")
        plt.plot(df2['r_E'].values[0],color='blue',linewidth=(i+1)*(j+1),label=rf"PYR $\alpha$={alpha},$\beta$={beta}",alpha=0.5)
        plt.plot(df2['r_I_PV'].values[0],color='green',linewidth=(i+1)*(j+1),label=rf"PV $\alpha$={alpha},$\beta$={beta}",alpha=(i+1)*(j+1)/3)
        plt.plot(df2['r_I_SOM'].values[0],color='red',linewidth=((i+1)*(j+1))/3,label=rf"SOM $\alpha$={alpha},$\beta$={beta}")

#changing order of legend
handles,labels=plt.gca().get_legend_handles_labels()
order=[0,3,6,1,4,7,2,5,8]
#legend outside the plot 
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)     
plt.xlabel("Time (ms)")
plt.ylabel(r"$r$ (Proportion of population firing)")
plt.title(r'Time series of $r$ for highly oscillatory values of $\alpha$ and $\beta$')

fig6.savefig('../results/figures/fig_6_dlx_firing_rates.png',format="png",bbox_inches='tight')

#now figure 7 which is power, freq


fig7=plt.figure(figsize=(10,10*3/4),dpi=500)
#want to show 3 beta values
alphas_=np.unique(np.linspace(coords['alpha'].min(),coords['alpha'].max(),3))
betas_=np.unique(np.linspace(coords['beta'].min(),coords['beta'].max(),3))
for i in range(0,len(alphas_)):
    for j in range(0,len(betas_)):

        alpha=alphas_[i]
        beta=betas_[j]
        df2=df.query(f"alpha=={alpha} & beta=={beta}& surr_size==0")
        plt.plot(df2['freqs_E'].values[0],df2['power_E'].values[0],color='blue',linewidth=(i+1)*(j+1),label=rf"PYR $\alpha$={alpha},$\beta$={beta}",alpha=0.5)
        plt.plot(df2['freqs_PV'].values[0],df2['power_PV'].values[0],color='green',linewidth=(i+1)*(j+1),label=rf"PV $\alpha$={alpha},$\beta$={beta}",alpha=(i+1)*(j+1)/3)
        plt.plot(df2['freqs_SOM'].values[0],df2['power_SOM'].values[0],color='red',linewidth=((i+1)*(j+1))/3,label=rf"SOM $\alpha$={alpha},$\beta$={beta}")

#changing order of legend
handles,labels=plt.gca().get_legend_handles_labels()
order=[0,3,6,1,4,7,2,5,8]
#legend outside the plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylim(pow(10,0),pow(10,4))
plt.xlim(0,100)
plt.yscale('log')

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Power (AU)")
plt.title(r'Power spectra of $r$ for highly oscillatory values of $\alpha$ and $\beta$')

fig7.savefig('../results/figures/fig_7_dlx_power_spectra.png',format="png",bbox_inches='tight')

#now figure 8 which is gamma power vs surround size
fig8=plt.figure(figsize=(10,10*3/4),dpi=500)
#want to show 3 alpha values
alphas_=np.unique(np.linspace(coords['alpha'].min(),coords['alpha'].max(),3))
betas_=np.unique(np.linspace(coords['beta'].min(),coords['beta'].max(),3))

for i in range(0,len(alphas_)):
    for j in range(0,len(betas_)):
        alpha=alphas_[i]
        beta=betas_[j]
        df2=df.query(f"alpha=={alpha} & beta=={beta} & surr_size<50")
        plt.plot(df2['surr_size'].values,df2['gamma_power'].values,color='blue',linewidth=(i+1)*(j+1),label=rf"$\alpha$={alpha},$\beta$={beta}",alpha=1)
    
# #changing order of legend
# handles,labels=plt.gca().get_legend_handles_labels()
# order=[0,3,6,1,4,7,2,5,8]
# #legend outside the plot
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.legend()
plt.xlabel("Surround size (AU)")
plt.ylabel(r"Gamma power (AU)")
plt.title(r'Gamma power vs surround size for highly oscillatory values of $\alpha$ and $\beta$')

fig8.savefig('../results/figures/fig_8_dlx_gamma_power.png',format="png",bbox_inches='tight')


#now that we are done with the dlx modelling we will show off the 'biologically informed model'



#first plot is gain functions


import params as prms
import numpy as np
import matplotlib.pyplot as plt
title_="Original gain function: \n"+r"$G(x)=\tanh(m_{Cell}*(x-\theta_{Cell}) \,\,\,\,\,  \{PYR,PV,SOM\} \in Cell$"
fig9=plot_Gain_function(xlim_list=[-20,20],pars=prms.default_pars(),annotation_height=0.4,PYR=prms.Gain_tanh_E,PV=prms.Gain_tanh_I_PV,SOM=prms.Gain_tanh_I_SOM,title=title_)
fig9.savefig("../results/figures/fig_9_gain_function_biological.png",format="png",bbox_inches='tight')


#now the final ting is the heatmap
import copy
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
from itertools import chain
import copy #for making deepcopies of 
#setting up initial parameters
connectivity_original=copy.deepcopy(prms.connectivity)
connectivity=connectivity_original

start_ab=0
stop_ab=30
alpha_range=np.around(np.linspace(start=start_ab,stop=stop_ab,num=11),2)
beta_range=np.around(np.linspace(start=start_ab,stop=stop_ab,num=11),2)
progress=0 #in percent
start=timer()

gain_functions=[prms.Gain_tanh]
connectivity_matrices=list(connectivity.keys())[2:3]


for _function in gain_functions:
    for connectivity_matrix_name in connectivity_matrices:
        gain_function=_function
        print("Doing gain function: ",_function.__name__,"with connectivity",connectivity_matrix_name)

        params_dict=prms.default_pars()


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
        connectivity_matrix_list=[]


        for i in list(alpha_range):
            alpha=i
            for j in list(beta_range):
                beta=j
                #updating alpha and beta
                scale=max(alpha_range)/2
                
                connectivity=copy.deepcopy(prms.connectivity)


                #W_EE
                connectivity[connectivity_matrix_name][0,0]*=1
                #W_I_PV_E
                connectivity[connectivity_matrix_name][0,1]*=alpha/scale
                #W_I_SOM_E
                connectivity[connectivity_matrix_name][0,2]*=beta/scale
                #W_E_I_PV
                connectivity[connectivity_matrix_name][1,0]*=alpha/scale
                #W_I_PV_I_PV
                connectivity[connectivity_matrix_name][1,1]*=alpha/ scale
                #W_I_SOM_I_PV
                connectivity[connectivity_matrix_name][1,2]*=0/scale
                #W_E_I_SOM
                connectivity[connectivity_matrix_name][2,0]*=beta/scale
                #W_I_PV_I_SOM
                connectivity[connectivity_matrix_name][2,1]*=beta/scale
                #W_I_SOM_I_SOM
                connectivity[connectivity_matrix_name][2,2]*=0/scale
                #W_E_I_L23
                connectivity[connectivity_matrix_name][3,0]*=0/scale
                #W_I_PV_E_L23
                connectivity[connectivity_matrix_name][3,1]*=alpha/scale
                #W_I_SOM_E_L23
                connectivity[connectivity_matrix_name][3,2]*=beta/scale
                #W_E_I_L4
                connectivity[connectivity_matrix_name][4,0]*=scale/scale
                #W_I_PV_E_L4
                connectivity[connectivity_matrix_name][4,1]*=alpha/scale
                #W_I_SOM_E_L4
                connectivity[connectivity_matrix_name][4,2]*=beta/scale

                #running the sim

                a=eqns.L234_E_PV_SOM_E(gain_func=gain_function,params=params_dict,surr_size=1,input_L4=1,connectivity_matrix=copy.deepcopy(connectivity[connectivity_matrix_name]))
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
                connectivity_matrix_list.append([connectivity_matrix_name,connectivity[connectivity_matrix_name]])
                end=timer()
                print("\nprogress\n",round(progress/((len(alpha_range))*len(beta_range)*len(gain_functions)*len(connectivity_matrices))*100,2),"%","\ntime_elapsed=\n",timedelta(seconds=end-start),"\nalpha=\n",alpha,"\nbeta=\n",beta,"\n\n")

    
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
        #adding Gain function inputs to the dataframe

        df['gain_input_exc']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_exc'])
        df['gain_input_pv']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_pv'])
        df['gain_input_som']=df['sim_data'].apply(lambda x: x['firing_rates']['Gain_Input_som'])
        df.to_csv(f"../results/alpha_beta_{min(alpha_range)}_{max(alpha_range)}_{_function.__name__}.tsv",sep="\t",encoding="utf-8")
        #df=pd.read_csv("../results/alpha_beta.csv")
        #visualising the range of gain function inputs:


        aggregate_gain_inputs=[]
        aggregate_input_E=[]
        aggregate_input_PV=[]
        aggregate_input_SOM=[]
        for pop in ['exc','pv','som']:
            col=df[f'gain_input_{pop}']
            for i in range(len(df)):
                aggregate_gain_inputs.append(df.iloc[i][f'gain_input_{pop}'])
                if pop=='exc':
                    aggregate_input_E.append(df.iloc[i][f'gain_input_{pop}'])
                elif pop=='pv':
                    aggregate_input_PV.append(df.iloc[i][f'gain_input_{pop}'])
                elif pop=='som':
                    aggregate_input_SOM.append(df.iloc[i][f'gain_input_{pop}'])

        aggregate_input_E=list(chain.from_iterable(aggregate_input_E))
        aggregate_input_PV=list(chain.from_iterable(aggregate_input_PV))
        aggregate_input_SOM=list(chain.from_iterable(aggregate_input_SOM))
        aggregate_gain_inputs=list(chain.from_iterable(aggregate_gain_inputs))

        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
        #%%
        fig10,axs =plt.subplots(3,4,figsize=(12,12*3/4),dpi=500,constrained_layout=True)

        df.drop_duplicates(['alpha','beta'],inplace=True)

        for i,pop in enumerate(["exc","pv","som"]):


            #whether oscillations are present or not
            pivot=df.pivot(index='alpha',columns='beta',values=f'total_power_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,0],annot_kws={"fontsize":8})
            axs[i,0].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,0].set_title(f"Total Power {pop.upper()} \n"+ r"Integral of FFT from $0$ to $100$ $Hz$")
            if pop=='PYR':
                pop='exc'
            axs[i,0].set_xlabel(r"$\beta$ (SOM)")
            axs[i,0].set_ylabel(r"$\alpha$ (PV)")

            
            pivot=df.pivot(index='alpha',columns='beta',values=f'avg_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,1],annot_kws={"fontsize":8},vmin=0,vmax=1.5)
            axs[i,1].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,1].set_title(f"{pop.upper()}\n"+r" Average $r$")
            if pop=='PYR':
                pop='exc'
            axs[i,1].set_xlabel(r"$\beta$ (SOM)")
            axs[i,1].set_ylabel(r"$\alpha$ (PV)")

            pivot=df.pivot(index='alpha',columns='beta',values=f'gain_input_{pop}')
            pivot=pivot.applymap(np.mean)
            sns.heatmap(pivot,annot=False,ax=axs[i,2],annot_kws={"fontsize":8},vmin=min(aggregate_gain_inputs),vmax=max(aggregate_gain_inputs))
            axs[i,2].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,2].set_title(f"{pop.upper()}\n Mean Synaptic Input")
            if pop=='PYR':
                pop='exc'
            axs[i,2].set_xlabel(r"$\beta$ (SOM)")
            axs[i,2].set_ylabel(r"$\alpha$ (PV)")

            
            
                
            pivot=df.pivot(index='alpha',columns='beta',values=f'n_peaks_{pop}')
            sns.heatmap(pivot,annot=False,ax=axs[i,3],annot_kws={"fontsize":8})
            axs[i,3].invert_yaxis()
            if pop=='exc':
                pop="PYR"
            axs[i,3].set_title(f"{pop.upper()}\n Number of peaks in FFT")
            if pop=='PYR':
                pop='exc'
            axs[i,3].set_xlabel(r"$\beta$ (SOM)")
            axs[i,3].set_ylabel(r"$\alpha$ (PV)")

        fig10.suptitle(r"using original Gain Function and biologically informed  connectivity matrix.",fontsize=12)
        #saving to file
        fig10.savefig(f"../results/figures/fig_10_alpha_beta_{min(alpha_range)}_{max(alpha_range)}_heatmap_general_{gain_function.__name__}_{connectivity_matrix_name}.jpg",format="jpg",dpi=500)
# %%
