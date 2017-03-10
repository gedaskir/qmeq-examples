# Prerequisites
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import qmeq

#---------------------------------------------------

# Quantum dot parameters
vgate, omega, U = 0.0, 2.0, 5.0
# Lead parameters
vbias, temp, dband = 0.5, 2.0, 60.0
# Tunneling amplitudes
gam = 1.0
t0 = np.sqrt(gam/(2*np.pi))

#---------------------------------------------------

nsingle = 2

hsingle = {(0,0): vgate,
           (1,1): vgate,
           (0,1): omega}

coulomb = {(0,1,1,0): U}

tleads = {(0,0): t0, # L <-- l
          (1,1): t0} # R <-- r

nleads = 2
#        L           R
mulst = {0: vbias/2, 1: -vbias/2}
tlst =  {0: temp,    1: temp}

system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband)

#---------------------------------------------------

def omega_vg(system, olst, vglst):
    opnt, vgpnt = olst.shape[0], olst.shape[0]
    mtr = np.zeros((opnt, vgpnt), dtype=float)
    for j1 in range(opnt):
        system.change(hsingle={(0,1):olst[j1]})
        for j2 in range(vgpnt):
            system.change(hsingle={(0,0):vglst[j2],
                                   (1,1):vglst[j2]})
            system.solve()
            mtr[j1, j2] = system.current[0]
    return mtr

def plot_omega_vg(mtr, olst, vglst, U, gam, kerntype, itype, num):
    (xmin, xmax, ymin, ymax) = (vglst[0]/U, vglst[-1]/U,
                                olst[0]/gam, olst[-1]/gam)
    p = plt.subplot(2, 2, num)
    p.set_xlabel('$V_{g}/U$', fontsize=20)
    p.set_ylabel('$\Omega/\Gamma$', fontsize=20)
    p.set_title(kerntype+', itype='+str(itype), fontsize=20)
    p_im = plt.imshow(mtr/gam, extent=[xmin, xmax, ymin, ymax],
                               vmin=0, vmax= 0.023,
                               aspect='auto',
                               origin='lower',
                               cmap=plt.get_cmap('Spectral'))
    cbar = plt.colorbar(p_im)
    cbar.set_label('Current [$\Gamma$]', fontsize=20)
    cbar.set_ticks(np.linspace(0.0, 0.03, 4))
    plt.tight_layout()

opnt, vgpnt = 201, 201
olst = np.linspace(0., 5., opnt)
vglst = np.linspace(-10., 10., vgpnt) - U/2

params = [['Pauli', 0, 1],
          ['1vN', 0, 2],
          ['1vN', 1, 3],
          ['1vN', 2, 4]]

fig = plt.figure(figsize=(12,8))
for kerntype, itype, num in params:
    system.kerntype = kerntype
    system.itype = itype
    mtr = omega_vg(system, olst, vglst)
    plot_omega_vg(mtr, olst, vglst, U, gam,
                  kerntype, itype, num)

plt.show()
fig.savefig('o_vg.pdf', bbox_inches='tight', dpi=100, pad_inches=0.0)
