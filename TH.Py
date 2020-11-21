import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

# simulation parameters
t0  = 0
tf  = 60
dt  = 0.001
N   = round((tf-t0)/dt)+1
t = np.array(range(N))*dt+t0

# Main data
P0      = 3000000   # [kW] - Nominal Power
deltaT  = 500       # [°C] - Difference between Tf and Tc in nominal condition
# ^ magari va rivisitato in base a proprietà dell'acqua
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

# input parameter initializzation
G       = G0*np.array([1.0 for i in range(N)])
Tin     = Tin0*np.array([1.0 for i in range(N)])
P       = P0*np.array([1.0 for i in range(N)])
I       = np.array([0])
# ^ links a passage in the integration step with the corrisponding passage
#   in the solver

def statedot(t,x):
# function that evaluates the evolution of coolant and fuel temperatures
# given P, Tin (coolant), G as parameters.
    # initializing time derivative
    z = [0.0 for i in range(2)]
    xdot = np.array(z)
    # compute time derivative
    xdot[0] = P[I[0]]/(Mf*Cf)-(x[0]-x[1])/tauf
    xdot[1] = (x[0]-x[1])/tauc-2*G[I[0]]/Mc*(x[1]-Tin[I[0]])
    return(xdot)

# dummy functions for the input
def G_step(t):
    if t<35:
        return G0
    else:
        return 1.1*G0

def Tin_step(t):
    if t<(tf+1):
        return Tin0
    else:
        return 1.1*Tin0

def P_step(t):
    if t<15:
        return P0
    else:
        return 1.1*P0

solver = ode(statedot)
solver.set_integrator('lsoda',max_step = dt/10)
# ^ here is important to set an upper limit for the integration step <dt
#   because, it is needed as sinc between the integrstion step an the
#   loading of the input functions
solver.set_initial_value(u0,t0)
# initializing solution
sol = [u0]

xdot = []
for i in range(0,N-1):
    if solver.successful() and solver.t < tf:
        I[0] = i+1
        G[I[0]] = G_step(t[I[0]])
        Tin[I[0]] = Tin_step(t[I[0]])
        P[I[0]] = P_step(t[I[0]])
        solver.integrate(t[I[0]])
        sol.append(solver.y.copy())
    else:
        break

sol = np.array(sol)
Tf = sol[:,0]
Tc = sol[:,1]
Tout = 2*Tc-Tin[range(len(Tc))]

plt.plot(t,P[range(len(t))])
plt.xlabel("Time [s]")
plt.ylabel("P [kW]")
plt.title("Power level")
plt.grid()
plt.show()

plt.plot(t,G[range(len(t))])
plt.xlabel("Time [s]")
plt.ylabel("G [kg/s]")
plt.title("Coolant mass flow rate level")
plt.grid()
plt.show()

plt.plot(t,Tf)
plt.xlabel("Time [s]")
plt.ylabel("Tf [°C]")
plt.title("Fuel mean temperature")
plt.grid()
plt.show()

plt.plot(t,Tc)
plt.xlabel("Time [s]")
plt.ylabel("Tc [°C]")
plt.title("Coolant mean temperature")
plt.grid()
plt.show()

plt.plot(t,Tout)
plt.xlabel("Time [s]")
plt.ylabel("Tout [°C]")
plt.title("Coolant outlet temperature")
plt.grid()
plt.show()

i = 0
while i < N and np.abs(Tf[i]-Tf0)<1e-5:
    i += 1
