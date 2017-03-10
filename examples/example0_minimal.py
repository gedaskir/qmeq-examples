# Prerequisites
import qmeq
from numpy import sqrt, pi, power

# Parameters used (numbers are in arbitrary energy_scale, e.g., meV)
e0, e1, omega, U = 0.0, 0.0, 0.0, 20.0
# dband is bandwidth D of the leads
tempL, tempR, muL, muR, dband= 1.0, 1.0 , 0.2, -0.2, 60.0
gammaL, gammaR = 0.5 , 0.7
tL, tR = sqrt(gammaL/(2*pi)), sqrt(gammaR/(2*pi))

## Hamiltonian, which requires definition

# nsingle is the number of single-particle states
# Here level 0 is spin up; level 1 is spin down
nsingle = 2
hsingle = {(0,0): e0, (1,1): e1, (0,1): omega}
# (0,1,1,0) represents an operator d0^{+}d1^{+}d1^{-}d0^{-}
coulomb = {(0,1,1,0): U}

## Lead and tunneling properties, which requires definition

# nleads is the number of lead channels
nleads = 4 # for two leads L/R with two spins up/down
# Here   L-up      R-up      L-down    R-down
mulst = {0: muL,   1: muR,   2: muL,   3: muR}
tlst =  {0: tempL, 1: tempR, 2: tempL, 3: tempR}

# The coupling matrix has indices(lead-spin, level)
tleads = {(0,0): tL, (1,0): tR, (2,1): tL, (3,1): tR}

## Construction of the transport system and
## calculation of currents into the system from all 4 channels

# Choice of approximate approach
# For kerntype='Redfield', '1vN', 'Lindblad', or 'Pauli'
system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Pauli')
system.solve()
print('Output is currents in four lead channels')
print('[L-up R-up L-down R-down]\n')
print('Pauli, particle current (in units of energy_scale/hbar):')
print(system.current)
print('Pauli, energy current (in units of energy_scale^2/hbar):')
print(system.energy_current)

# For '2vN' approach
# Here we use 2^12 energy grid points for the leads
kpnt = power(2,12)
system2vN = qmeq.Builder(nsingle, hsingle, coulomb,
                         nleads, tleads, mulst, tlst, dband,
                         kerntype='2vN', kpnt=kpnt)
# 7 iterations is good for temp approx gamma
system2vN.solve(niter=7)
print('\n2vN, particle current (in units of energy_scale/hbar):')
print(system2vN.current)
print('2vN, energy current (in units of energy_scale^2/hbar):')
print(system2vN.energy_current)
