# Prerequisites
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import qmeq

#---------------------------------------------------

# Quantum dot parameters
vgate, bfield, omega, U = 0.0, 0.0, 0.0, 20.0
# Lead parameters
vbias, temp, dband = 0.0, 1.0, 60.0
# Tunneling amplitudes
gam = 0.5
t0 = np.sqrt(gam/(2*np.pi))

#---------------------------------------------------

nsingle = 2
# 0 is up, 1 is down
hsingle = {(0,0): vgate+bfield/2,
           (1,1): vgate-bfield/2,
           (0,1): omega}

coulomb = {(0,1,1,0): U}

tleads = {(0,0): t0, # L, up   <-- up
          (1,0): t0, # R, up   <-- up
          (2,1): t0, # L, down <-- down
          (3,1): t0} # R, down <-- down
                     # lead label, lead spin <-- level spin

nleads = 4
#        L,up        R,up         L,down      R,down
mulst = {0: vbias/2, 1: -vbias/2, 2: vbias/2, 3: -vbias/2}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp}

system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Pauli')

kpnt = np.power(2,12)
system2vN = qmeq.Builder(nsingle, hsingle, coulomb,
                         nleads, tleads, mulst, tlst, dband,
                         kerntype='2vN', kpnt=kpnt)

def trace_vbias(system, vlst, vgate, bfield, dV=0.01, niter=7):
    print(system.kerntype)
    vpnt = vlst.shape[0]
    trace = np.zeros(vpnt)
    #
    system.change(hsingle={(0,0): vgate+bfield/2,
                           (1,1): vgate-bfield/2})
    system.solve(masterq=False)
    #
    for j1 in range(vpnt):
        system.change(mulst={0: vlst[j1]/2, 1: -vlst[j1]/2,
                             2: vlst[j1]/2, 3: -vlst[j1]/2})
        system.solve(qdq=False, niter=niter)
        trace[j1] = (system.current[0]
                   + system.current[2])
        #
        system.add(mulst={0: dV/2, 1: -dV/2,
                          2: dV/2, 3: -dV/2})
        system.solve(qdq=False, niter=niter)
        trace[j1] = (system.current[0]
                   + system.current[2]
                   - trace[j1])/dV
        print(vlst[j1], trace[j1]/gam)
    return trace

vpnt = 101
vlst = np.linspace(0, 2*U, vpnt)
trace_Pauli = trace_vbias(system, vlst, -U/2, 7.5)
trace_2vN = trace_vbias(system2vN, vlst, -U/2, 7.5)
np.savetxt('trace_2vN.dat', trace_2vN)

fig = plt.figure()
p = plt.subplot(1, 1, 1)
p.set_xlabel('$V/U$', fontsize=20)
p.set_ylabel('$\mathrm{d}I/\mathrm{d}V$', fontsize=20)
plt.plot(vlst/U, trace_Pauli, label='Pauli',
                              color='blue',
                              lw=2)
plt.plot(vlst/U, trace_2vN, label='2vN',
                            color='black',
                            lw=2,
                            linestyle='--')
plt.legend(loc=2, fontsize=20)
plt.show()
fig.savefig('Pauli_vs_2vN.pdf', bbox_inches='tight',
                                dpi=100,
                                pad_inches=0.0)
