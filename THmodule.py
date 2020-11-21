import numpy as np

# TH main data
def THdataBuilder():
    P0      = 3000000   # [kW] - Nominal Power
    deltaT  = 50        # [°C] - Difference between Tf and Tc in nominal condition
    Tin0    = 280       # [°C] - Coolant inlet temperature in nominal condition
    Mf      = 100000    # [kg] - Fuel mass
    Cf      = 0.35      # [kJ/(kg*K)] - Fuel specific heat capacity
    Mc      = 13000     # [kg] - Coolant mass
    Cc      = 5.5       # [kJ/(kg*K)] - Coolant specific heat capacity
    G0      = 10000     # [kg/s] - Mass flow rate
    K       = P0/deltaT # [kW/°C]
    tauf    = Mf*Cf/K   # [s]
    tauc    = Mc*Cc/K   # [s]
    # nominal value for state variable
    Tc0     = Tin0+P0/(2*G0*Cc)
    Tf0     = P0/K+Tc0
    u0      = np.array([Tf0,Tc0])
    return (P0,Tin0,G0,Mf,Cf,Mc,Cc,K,tauf,tauc,u0)

def THdot(t,x,param):
    # Funcion: x_dot = fun(t,x,param) that returns the first derivative of
    # the system state x evaluated at time t given the following param structure
    # param = [ [Power level (array)],
    #           [Water Inlet Temperature (array)],
    #           [Water Mass flow Rate (array)],
    #           [Fuel Mass (float)],
    #           [six group dacay constant (array)],
    #           [six group resp. fractions (array)]     ]
    P = param[0]
    Tin = param[1]
    G = param[2]
    Mf = param[3]
    Cf = param[4]
    Mc = param[5]
    tauc = param[6]
    A = param[7]
    p = param[-1]
    # initializing time derivative
    x_dot = np.array([0.0,0.0])
    # input state-indipendent contribution
    Ft = np.array([P[p[0]]/(Mf*Cf),2*G[p[0]]/Mc*Tin[p[0]]])
    # load the input dependent part of A
    A[1,1] = -(1/tauc+2*G[p[0]]/Mc)
    # compute time derivative
    for i in range(len(x_dot)):
        x_dot[i] = Ft[i] + np.dot(A[i, :], np.real(x.transpose()))
    return(x_dot)

#########################################################################
# step function definition for TH input #################################
# mass flow rate step
def G_step(t,G0,dlogG=1.1,T=np.inf):
    if t<T:
        return G0
    else:
        return dlogG*G0
# Inlet temperature step
def Tin_step(t,Tin0,dlogTin=1.1,T=np.inf):
    if t<T:
        return Tin0
    else:
        return dlogTin*Tin0
# Power step
def P_step(t,P0,dlogP=1.1,T=1):
    if t<T:
        return P0
    else:
        return dlogP*P0
