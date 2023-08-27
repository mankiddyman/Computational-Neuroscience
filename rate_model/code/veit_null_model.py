    #here I am going to generate the exact same model as in veit et al and produce the following figures
    #first a 2 panel simulation with rates and ffts of all populations
    #second I will reproduce the gamma- surround relationship in a 2 panel plot of gp vs surr and fft spectra for different surr values
    #third I will create alpha beta plot?
    #fourth I will decide how to model the dlx mutants


    import numpy as np
    import matplotlib.pyplot as plt
    import eqns
    import params 
    from params import params_dict_original
    from types import SimpleNamespace

    #generate a simulation
    #set up the parameters
    params_dict = params_dict_original

    sim=eqns.L234_E_PV_SOM_E(gain_func=params.Gain_veit,params=params_dict,input_L4=1,surr_size=1,add_noise=0.000,connectivity_matrix=params.connectivity['veit'])
    s=SimpleNamespace(**sim['firing_rates'])
    #now I want fft spectra of all 3 populations
    #first calculating for the excitatory population
    r_E=s.r_E_L23
    x0=r_E[int(0.3*len(r_E)):]#skipping the first 30% of the data
    power_exc,freqs_exc=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="Exc",plotting=False)
    #now calculating for the PV population
    r_I_PV=s.r_I_PV
    x0=r_I_PV[int(0.3*len(r_I_PV)):]#skipping the first 10% of the data
    power_pv,freqs_pv=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="PV",plotting=False)
    #now calculating for the SOM population
    r_I_SOM=s.r_I_SOM
    x0=r_I_SOM[int(0.3*len(r_I_SOM)):]#skipping the first 10% of the data
    power_som,freqs_som=eqns.FFT_updated(x0=x0,dt=sim['params']['dt'],T_end=sim['params']['T_end'],title="SOM",plotting=False)

    power_exc,power_pv,power_som=np.abs(power_exc),np.abs(power_pv),np.abs(power_som)

    #now make plot 1
    #%%
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(s.Time_series,s.r_E_L23,label="E",color='blue')
    ax[0].plot(s.Time_series,s.r_I_SOM,label="SOM",color='red')
    ax[0].plot(s.Time_series,s.r_I_PV,label="PV",color='green')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Firing Rate (Hz)")
    #add legend
    ax[0].legend(bbox_to_anchor=(.95, 1), loc='lower right', borderaxespad=0.,title="Cell Type")
    #so somatostatin doesn't normally oscillate in this model without noise
    ax[1].set_yscale('log')
    ax[1].plot(freqs_exc,power_exc,label="E",color='blue')
    ax[1].plot(freqs_pv,power_pv,label="PV",color='green')
    ax[1].plot(freqs_som,power_som,label="SOM",color='red')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Power")
    ax[1].set_xlim(0,200)
    ax[1].set_ylim(1e0,1e5)


    # %%

