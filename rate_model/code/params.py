#following params (at least at the beginning were taken from here https://www.nature.com/articles/nn.4562#Sec4 suplementary table 1)
params_dict={}
params_dict['dt']=0.01
params_dict['T_end']=1000

#E_L23
params_dict['tau_E']=20 #ms 
params_dict['theta_E']=-11 #threshold input for pop E
params_dict['m_E']=.25#rate of response for pop_E
params_dict['W_EE']=16
params_dict['W_E_I_PV']=26
params_dict['W_E_I_SOM']=15
params_dict['W_EE_L4']=1 #strangely not mentioned in suplementary table 1
params_dict['W_EE_L23']=0 #this is a reccurent connection within L2/3
params_dict['MIN_i_E_L23']= 1.4 #minimum value of input when stim is there?
params_dict['m_i_E_L23']=0.2 #how fast input increases with surround size
params_dict['exp_E']=1 #exponent for E_L23

#I_SOM
params_dict['tau_I_SOM']=10 #ms #they keep them constant for both SOM and PV
params_dict['theta_I_SOM']=0.65 
params_dict['m_I_SOM']=0.1
params_dict['W_I_SOM_E']=0.25
params_dict['W_I_SOM_I_PV']=0
params_dict['W_I_SOM_I_SOM']=0.025
params_dict['W_I_SOM_E_L4']=0
params_dict['W_I_SOM_E_L23']=2
params_dict['exp_I_SOM']=1 #added myself, not in veit et al 2017

#I_PV
params_dict['tau_I_PV']=10 #ms
params_dict['theta_I_PV']=13
params_dict['m_I_PV']=0.005
params_dict['W_I_PV_E']=20
params_dict['W_I_PV_I_PV']=1
params_dict['W_I_PV_I_SOM']=1
params_dict['W_I_PV_E_L4']=1
params_dict['W_I_PV_E_L23']=0.5
params_dict['exp_I_PV']=3


params_dict_original=params_dict.copy() # identical to Veit et al 2017 parameters (suplementary table 1) https://www.nature.com/articles/nn.3446#Fig7

params_dict_pfeffer_2013=params_dict_original.copy() #this copy involves the parameters from Pfeffer et al 2013 https://www.nature.com/articles/nn.3446#Fig7 as I understand them, strangely veit et al 2017 does not incoroporate them in a clear manner. 
#I will also be changing the weights for L4 and L2/3 to values that make sense to me

#beginning with L4
params_dict_pfeffer_2013['W_EE_L4']=1 #set to 1 because L4 is the input layer which inputs to Pyramidal cells in L2/3
params_dict_pfeffer_2013['W_I_SOM_E_L4']=0 #set to 0 because L4 does not synapse onto SOM cells in L2/3
params_dict_pfeffer_2013['W_I_PV_E_L4']=1 #set to 1 because L4 does synapse onto PV cells in L2/3

#now L2/3
params_dict_pfeffer_2013['W_EE_L23']=0.5 #set to 0 because Horizontal  L2/3 connections do synapse onto pyramidal cells in L2/3 (as given by figure 3 in veit et al 2017). The value is chosen to be equal to W_I_PV_E_L23
params_dict_pfeffer_2013['W_I_PV_E_L23']=0.5 #see above NB identical to params_dict_orignal
params_dict_pfeffer_2013['W_I_SOM_E_L23']=2 #Identical to params_dict_orignal

#now the weights between populations in L2/3
params_dict_pfeffer_2013['W_EE']=1 
params_dict_pfeffer_2013['W_E_I_PV']=1
params_dict_pfeffer_2013['W_E_I_SOM']=0.54

params_dict_pfeffer_2013['W_I_SOM_E']=1 #assuming exc excites everything equally
params_dict_pfeffer_2013['W_I_SOM_I_PV']=0.03
params_dict_pfeffer_2013['W_I_SOM_I_SOM']=0.02

params_dict_pfeffer_2013['W_I_PV_E']=1 #assuming exc excites everything equally
params_dict_pfeffer_2013['W_I_PV_I_PV']=1
params_dict_pfeffer_2013['W_I_PV_I_SOM']=0.33

#now we have two dictionaries

params_dict_aaryan = params_dict_original.copy() #params with biologically informed weights and all other things kept equal where possible

#first designing the population activation function. theta and m are kept the same for all populations
params_dict_aaryan['theta_E'], params_dict_aaryan['m_E'] = 10, 0.005
params_dict_aaryan['theta_I_SOM'], params_dict_aaryan['m_I_SOM'] = 10, 0.005
params_dict_aaryan['theta_I_PV'], params_dict_aaryan['m_I_PV'] = 10, 0.005

#now the exponent
params_dict_aaryan['exp_E'], params_dict_aaryan['exp_I_SOM'], params_dict_aaryan['exp_I_PV'] = 3, 3, 3


alpha=20
beta=20
#now the weights
params_dict_aaryan['W_EE']=50
params_dict_aaryan['W_E_I_PV']=alpha
params_dict_aaryan['W_E_I_SOM']=beta

params_dict_aaryan['W_I_PV_E']=alpha
params_dict_aaryan['W_I_PV_I_PV']=alpha  
params_dict_aaryan['W_I_PV_I_SOM']=beta/2  

params_dict_aaryan['W_I_SOM_E']=beta
params_dict_aaryan['W_I_SOM_I_PV']=0.0
params_dict_aaryan['W_I_SOM_I_SOM']=0.01

