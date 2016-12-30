# Prerequisites
import qmeq
from numpy import sqrt, pi, power

# Parameters used (numbers are in arbitrary energy scale, e.g., meV)
e0, e1, U = 0.0, 0.0, 20.0
tempL, tempR, muL, muR, dband= 1.0, 1.0 , 0.2, -0.2, 60.0
gammaL, gammaR = 0.5 , 0.7
tL, tR = sqrt(gammaL/(2*pi)), sqrt(gammaR/(2*pi))

## Hamiltonian, which requires definition

nsingle = 2 # Here level 0 is spin up; level 1 is spin down
hsingle = {(0,0): e0, (1,1): e1}
coulomb = {(0,1,1,0): U}

## Lead and tunneling properties, which requires definition

nleads = 4 # for two leads L/R with two spins up/down
# Here   L-up      R-up      L-down    R-down
mulst = {0: muL,   1: muR,   2: muL,   3: muR}
tlst =  {0: tempL, 1: tempR, 2: tempL, 3: tempR}

# The coupling matrix has indices(lead-spin, level)
tleads = {(0,0): tL, (1,0): tR, (2,1): tL, (3,1): tR}

## Construction of the transport system and
## calculation of currents into the system from all 4 channels

# For kerntype='Redfield', '1vN', 'Lindblad', or 'Pauli'
system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Pauli')
system.solve()
print(system.current) # particle current [Energy scale/hbar]
print(system.energy_current) # energy current [Energy scale^2/hbar]

# For '2vN' approach
kpnt = power(2,12) # Here we use 2^12 points for the leads
system2vN = qmeq.Builder(nsingle, hsingle, coulomb,
                         nleads, tleads, mulst, tlst, dband,
                         kerntype='2vN', kpnt=kpnt)
system2vN.solve(niter=7) # 7 interations is good for temp approx gamma
print(system2vN.current) # particle current [energy scale/hbar]
print(system2vN.energy_current) # energy current [energy scale^2/hbar]
