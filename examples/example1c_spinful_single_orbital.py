# Prerequisites
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import qmeq

#---------------------------------------------------

def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    #
    for j1 in range(vgpnt):
        system.change(hsingle={(0,0):vglst[j1]+bfield/2,
                               (1,1):vglst[j1]-bfield/2})
        system.solve(masterq=False)
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2,
                                 2: vlst[j2]/2, 3: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = (system.current[0]
                          + system.current[2])
            #
            system.add(mulst={0: dV/2, 1: -dV/2,
                              2: dV/2, 3: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0]
                               + system.current[2]
                               - stab[j1, j2])/dV
    #
    return stab, stab_cond

def stab_plot(stab, stab_cond, vlst, vglst, U, gam, fname='fig.pdf'):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1],
                                         vlst[0], vlst[-1]])/U
    fig = plt.figure(figsize=(12,4.2))
    #
    p1 = plt.subplot(1, 2, 1)
    p1.set_xlabel('$V_{g}/U$', fontsize=20)
    p1.set_ylabel('$V/U$', fontsize=20)
    p1_im = plt.imshow(stab.T/gam, extent=[xmin, xmax, ymin, ymax],
                                   aspect='auto',
                                   origin='lower',
                                   cmap=plt.get_cmap('Spectral'))
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [$\Gamma$]', fontsize=20)
    #
    p2 = plt.subplot(1, 2, 2)
    p2.set_xlabel('$V_{g}/U$', fontsize=20);
    p2.set_ylabel('$V/U$', fontsize=20);
    p2_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax],
                                    aspect='auto',
                                    origin='lower',
                                    cmap=plt.get_cmap('Spectral'))
    cbar2 = plt.colorbar(p2_im)
    cbar2.set_label('Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    #
    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.show()

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

kpnt = np.power(2,12)
system2vN = qmeq.Builder(nsingle, hsingle, coulomb,
                         nleads, tleads, mulst, tlst, dband,
                         kerntype='2vN', kpnt=kpnt)

vpnt, vgpnt = 201, 201
vlst = np.linspace(-2*U, 2*U, vpnt)
vglst = np.linspace(-2.5*U, 1.5*U, vgpnt)
stab_b_2vN, stab_cond_b_2vN = stab_calc(system2vN, 7.5, vlst, vglst)
stab_plot(stab_b_2vN, stab_cond_b_2vN, vlst, vglst, U, gam, 'stab2vN.pdf')
