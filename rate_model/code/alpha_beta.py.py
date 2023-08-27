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


        #final pivot plot

        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
        #%%
        fig1,axs =plt.subplots(3,7,figsize=(20,10),dpi=1000,constrained_layout=True)

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
            sns.heatmap(pivot,annot=False,ax=axs[i,5],annot_kws={"fontsize":8},vmin=0,vmax=1.5)
            axs[i,5].invert_yaxis()
            axs[i,5].set_title(f"avg_{pop}")
            axs[i,5].set_xlabel("Beta (SOM)")
            axs[i,5].set_ylabel("Alpha (PV)")

            pivot=df.pivot(index='alpha',columns='beta',values=f'gain_input_{pop}')
            pivot=pivot.applymap(np.mean)
            sns.heatmap(pivot,annot=False,ax=axs[i,6],annot_kws={"fontsize":8},vmin=min(aggregate_gain_inputs),vmax=max(aggregate_gain_inputs))
            axs[i,6].invert_yaxis()

            axs[i,6].set_title(f"mean_gain_input_{pop}")
            axs[i,6].set_xlabel("Beta (SOM)")
            axs[i,6].set_ylabel("Alpha (PV)")


        fig1.suptitle(f"Gain Function: {gain_function.__name__} Connectivity_matrix: {connectivity_matrix_name}")
        #saving to file
        fig1.savefig(f"../results/alpha_beta_{min(alpha_range)}_{max(alpha_range)}_heatmap_general_{gain_function.__name__}_{connectivity_matrix_name}.jpg",format="jpg",dpi=750)
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
        fig2.savefig(f"../results/alpha_beta_{min(alpha_range)}_{max(alpha_range)}_{gain_function.__name__}_{connectivity_matrix_name}.jpg",format="jpg",dpi=750)

        # %%
        # alph=70
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")


        # # #visualising the range of gain function inputs:
        # aggregate_gain_inputs=[]
        # aggregate_input_E=[]
        # aggregate_input_PV=[]
        # aggregate_input_SOM=[]
        # for pop in ['E','PV','SOM']:

        #     col=df[f'gain_input_{pop}']
        #     for i in range(len(df)):
            
        #         aggregate_gain_inputs.append(df.iloc[i][f'gain_input_{pop}'])
        #         if pop=='E':
        #             aggregate_input_E.append(df.iloc[i][f'gain_input_{pop}'])
        #         elif pop=='PV':
        #             aggregate_input_PV.append(df.iloc[i][f'gain_input_{pop}'])
        #         elif pop=='SOM':
        #             aggregate_input_SOM.append(df.iloc[i][f'gain_input_{pop}'])

        # aggregate_input_E=list(chain.from_iterable(aggregate_input_E))
        # aggregate_input_PV=list(chain.from_iterable(aggregate_input_PV))
        # aggregate_input_SOM=list(chain.from_iterable(aggregate_input_SOM))
        # aggregate_gain_inputs=list(chain.from_iterable(aggregate_gain_inputs))

        #%%
        fig3,axes=plt.subplots(2,2,figsize=(10,10),dpi=500,constrained_layout=True)

        ax1,ax2,ax3,ax4=axes.flatten()

        mean_exc=np.mean(aggregate_input_E)
        mean_pv=np.mean(aggregate_input_PV)
        mean_som=np.mean(aggregate_input_SOM)

        n_bins=30
        sns.histplot(aggregate_gain_inputs,bins=n_bins,ax=ax1)
        ax1.set_xlabel("Gain function input")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Histogram of all population gain inputs for {gain_function.__name__} with connectivity {connectivity_matrix_name}")
        ax1.xlim=(min(aggregate_gain_inputs),max(aggregate_gain_inputs))
        #add a line for the mean of each population
        ax1.axvline(mean_exc,color='blue',label=f"Mean exc={mean_exc}")
        ax1.axvline(mean_pv,color='green',label=f"Mean pv={mean_pv}")
        ax1.axvline(mean_som,color='red',label=f"Mean som={mean_som}")
        #adding as legend with big fontsize

        fig3.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=20)



        sns.histplot(aggregate_input_E,bins=n_bins,ax=ax2,color='blue')
        ax2.set_xlabel("Gain function input")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Histogram of Inputs to EXC {gain_function.__name__}")
        ax2.xlim=(min(aggregate_gain_inputs),max(aggregate_gain_inputs))

        sns.histplot(aggregate_input_PV,bins=n_bins,ax=ax3,color='green')
        ax3.set_xlabel("Gain function input")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"Histogram of Inputs to PV {gain_function.__name__}")
        ax3.xlim=(min(aggregate_gain_inputs),max(aggregate_gain_inputs))

        sns.histplot(aggregate_input_SOM,bins=n_bins,ax=ax4,color='red')
        ax4.set_xlabel("Gain function input")
        ax4.set_ylabel("Frequency")
        ax4.set_title(f"Histogram of Inputs to SOM {gain_function.__name__}")
        ax4.xlim=(min(aggregate_gain_inputs),max(aggregate_gain_inputs))
        fig3.savefig(f"../results/alpha_beta_{min(alpha_range)}_{max(alpha_range)}_{gain_function.__name__}_{connectivity_matrix_name}_input_histogram.jpg",format="jpg",dpi=500)

        # del df

        #%%

        # %%
        # alph=60
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

        # alph=50
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")

        # alph=40
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n High PV and low SOM")


        # alph=80
        # bet=20
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n Higher PV and low SOM \n (SOM fixation over inhibiting PV and constant EXC)")

        # alph=10
        # bet=80
        # visualise_sim(df.query(f"alpha=={alph} and beta=={bet}")['sim_data'].values[0],title=f"alpha={alph},beta={bet} \n Higher PV and low SOM \n (SOM fixation over inhibiting PV and constant EXC)")

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
