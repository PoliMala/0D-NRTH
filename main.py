import numpy as np
from scipy.integrate import ode
import os
#  set the 'MPLCONFIGDIR' to '/tmp' to get a writable environment for matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib.pyplot as plt
import csv
from THmodule import *

################################################################################
# simulation parameters ########################################################
# time discretizzation
t0  = 0
tf  = 60
dt  = 0.05
N   = round((tf-t0)/dt)+1
t = np.array(range(N))*dt+t0
# time index init
I = np.array([0])
# data loading
[P0,Tin0,G0,Mf,Cf,Mc,Cc,K,tauf,tauc,u0] = THdataBuilder()
# input/state variable initializing
P = P0*np.array([1.0 for ii in range(N)])
G = G0*np.array([1.0 for ii in range(N)])
Tin = Tin0*np.array([1.0 for ii in range(N)])
# statedot matrix (input dependent) init
A = np.array([[-1/tauf,1/tauf],[1/tauc,0.0]])
# TH statedot parameters (see THmodule)
params = [P,Tin,G,Mf,Cf,Mc,tauc,A,I]
solver = ode(THdot)
solver.set_integrator('lsoda',max_step = dt/7)
solver.set_initial_value(u0,t0).set_f_params(params)
# initializing solution
sol = [u0]

DN = 100 # number of steps between two messages
nsteps = DN
print('------------------------------')
# SOLVE THE PROBLEM
for ii in range(0,N-1):
    # this tells us how many steps have been performed
    if (ii / nsteps == 1):
        print('------------------------------')
        print(str(nsteps), ' steps performed out of ',str(N-1))
        nsteps = nsteps + DN
        print('------------------------------')
    if solver.successful() and solver.t < tf:
        I[0] = ii+1
        G[I[0]] = G_step(t[I[0]],G0)
        Tin[I[0]] = Tin_step(t[I[0]],Tin0)
        P[I[0]] = P_step(t[I[0]],P0)
        solver.integrate(t[I[0]])
        sol.append(np.real(solver.y.copy()))
    else:
        break
print('------------------------------')
print('Simulation Completed')
print('------------------------------')
print('------------------------------')

################################################################################
# postprocessing ###############################################################
sol = np.array(sol)
Tf = sol[:,0]
Tc = sol[:,1]
Tout = 2*Tc-Tin[range(len(Tc))]

plt.figure(1)
plt.plot(t,P[range(len(t))])
plt.xlabel("Time [s]")
plt.ylabel("Power Level [kW]")
plt.title("Power input time profile")
plt.grid()
plt.savefig('THPt.png')

plt.figure(2)
plt.plot(t,Tf)
plt.xlabel("Time [s]")
plt.ylabel("Fuel Temperature [°C]")
plt.title("Fuel Temperature time profile")
plt.grid()
plt.savefig('THTft.png')

plt.figure(3)
plt.plot(t,Tc)
plt.xlabel("Time [s]")
plt.ylabel("Coolant Temperature [°C]")
plt.title("Coolant Temperature time profile")
plt.grid()
plt.savefig('THTct.png')

# to show all figures
#plt.show()

# save a csv file
solFileName = "THsol.csv"
with open(solFileName, 'w') as solFile:
  writer = csv.writer(solFile)
  rowNum = 0
  writer.writerow(['Time [s]', 'Power [W]', 'Cool. Mass Flow Rate [kg/s]','Cool. in Temp. [°C]','Fuel Temp. [°C]','Cool. Temp. [°C]'])
  writer.writerows([[t[jj],P[jj],G[jj],Tin[jj],Tf[jj],Tc[jj]] for jj in range(len(t))])
