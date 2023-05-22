#following params (at least at the beginning were taken from here https://www.nature.com/articles/nn.4562#Sec4 suplementary table 1)
params_dict={}
params_dict['dt']=0.1
params_dict['T_end']=500

#E_L23
params_dict['tau_E']=20 #ms 
params_dict['theta_E']=-11 #threshold input for pop E
params_dict['m_E']=.25#rate of response for pop_E
params_dict['W_EE']=16
params_dict['W_E_I_PV']=26
params_dict['W_E_I_SOM']=15
params_dict['W_EE_L4']=1 #strangely not mentioned in suplementary table 1
params_dict['W_EE_L23']=0
params_dict['MIN_i_E_L23']= 1.4 #minimum value of input when stim is there?
params_dict['m_i_E_L23']=0.2 #how fast input increases with surround size

#I_SOM
params_dict['tau_I_SOM']=10 #ms #they keep them constant for both SOM and PV
params_dict['theta_I_SOM']=0.65 
params_dict['m_I_SOM']=0.1
params_dict['W_I_SOM_E']=0.25
params_dict['W_I_SOM_I_PV']=0
params_dict['W_I_SOM_I_SOM']=0.025
params_dict['W_I_SOM_E_L4']=1
params_dict['W_I_SOM_E_L23']=2


#I_PV
params_dict['tau_I_PV']=10 #ms
params_dict['theta_I_PV']=13
params_dict['m_I_PV']=0.005
params_dict['W_I_PV_E']=20
params_dict['W_I_PV_I_PV']=1
params_dict['W_I_PV_I_SOM']=1
params_dict['W_I_PV_E_L4']=1
params_dict['W_I_PV_E_L23']=0.5

