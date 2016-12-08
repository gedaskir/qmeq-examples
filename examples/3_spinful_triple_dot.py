# Prerequisites
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import qmeq
import itertools

#---------------------------------------------------

# Lead and tunneling parameters
mul, mur = 50.0, 10.0
gam, temp, dband = 0.1, 1.0, np.power(10.0, 4)

tl, tr = np.sqrt(gam/(2*np.pi)), np.sqrt(gam/(2*np.pi))
tleads = {(0,0): +tl, (0,1): +tl, (1,4): -tr,
          (2,5): +tl, (2,6): +tl, (3,9): -tr}

nleads = 4
mulst = {0: mul,   1: mur,   2: mul,   3: mur}
tlst  = {0: temp,  1: temp,  2: temp,  3: temp}

#---------------------------------------------------

# Quantum dot single-particle Hamiltonian
nsingle = 10
norb = nsingle//2
e0, e1, e2, e3, e4 = 60, 40, 38, 20, 20
o02, o03, o12, o13, o24, o34 = 0.2, 0.1, 0.1, -0.05, 0.2, 0.1

# Spin up Hamiltonian
hsingle0 = np.array([[e0,  0,   o02, o03, 0],
                     [0,   e1,  o12, o13, 0],
                     [o02, o12, e2,  0,   o24],
                     [o03, o13, 0,   e3,  o34],
                     [0,   0,   o24, o34, e4]])
# Augment the Hamiltonian to have spin up and down
hsingle = np.kron(np.eye(2), hsingle0)

#---------------------------------------------------

# Coulom matrix elements
usc = -0.1
dotindex = [0, 0, 1, 1, 2, 0, 0, 1, 1, 2]
coulomb = {}
for m, n, k, l in itertools.product(range(nsingle), repeat=4):
    if m!=n and k!=l and m//norb==l//norb and n//norb==k//norb:
        # Interdot iteraction
        # Note that the pairs (n,k) and (m,l) are at different dots
        if (dotindex[m] == dotindex[l] and
            dotindex[n] == dotindex[k] and
            abs(dotindex[m]-dotindex[n]) == 1):
            if n != k and m != l:
                # Charge-quadrupole
                coulomb.update({(m,n,k,l): usc})

#---------------------------------------------------

system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      indexing='ssq', kerntype='Pauli', itype=2)

def trace_e3(system, e3lst, removeq=False, dE=150.0):
    print(system.kerntype)
    e3pnt = e3lst.shape[0]
    trace = np.zeros(e3pnt)
    system.use_all_states()
    for j1 in range(e3pnt):
        system.change({(3, 3): e3lst[j1],
                       (8, 8): e3lst[j1]})
        system.solve(masterq=False)
        if removeq:
            system.remove_states(dE)
        system.solve(qdq=False)
        trace[j1] = sum(system.current[np.ix_([0,2])])
        print(e3lst[j1], trace[j1]/gam)
    return trace

e3pnt = 201
e3lst = np.linspace(15., 25., e3pnt)

system.kerntype = 'Pauli'
tr_e3_Pauli = trace_e3(system, e3lst)
np.savetxt('tr_e3_Pauli.dat', tr_e3_Pauli)
tr_e3_Pauli_rm = trace_e3(system, e3lst, removeq=True)
np.savetxt('tr_e3_Pauli_rm.dat', tr_e3_Pauli_rm)

system.kerntype = '1vN'
"""
1vN calculation with all the states included requires a lot of memory (~6GB)
Change False to True below if you want to perform this calculation.
"""
tr_e3_1vN = trace_e3(system, e3lst) if False else np.zeros(e3pnt)
np.savetxt('tr_e3_1vN.dat', tr_e3_1vN)
tr_e3_1vN_rm = trace_e3(system, e3lst, removeq=True)
np.savetxt('tr_e3_1vN_rm.dat', tr_e3_1vN_rm)

fig = plt.figure()
p = plt.subplot(1, 1, 1)
p.set_xlabel('$E_{3}/T$', fontsize=20)
p.set_ylabel('Current [$\Gamma$]', fontsize=20)
plt.plot(e3lst/temp, tr_e3_Pauli/gam,
         label='Pauli', color='blue', lw=2)
plt.plot(e3lst/temp, tr_e3_Pauli_rm/gam,
         label='Pauli (reduced)', color='blue', lw=1)
plt.plot(e3lst/temp, tr_e3_1vN/gam,
         label='1vN', color='black', lw=2, linestyle='--')
plt.plot(e3lst/temp, tr_e3_1vN_rm/gam,
         label='1vN  (reduced)', color='black', lw=1, linestyle='--')
plt.legend(loc=1, fontsize=20)
plt.show()
fig.savefig('Pauli_vs_1vN.pdf', bbox_inches='tight',
                                dpi=100,
                                pad_inches=0.0)
