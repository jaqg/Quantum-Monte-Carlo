from IO import title, section, read_input, read_xyz, print_input, end
from debug import all_debug
from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi, potential_ee, potential_eN, potential_NN, potential, kinetic_e, kinetic_N, kinetic, e_loc
from MonteCarlo import average, error, MC, Metropolis_symmetric_MC, Metropolis_generalized_MC, Pure_diffusion_MC, VMC, output_VMC, PDMC, output_PDMC

import numpy as np

# =============================================================================
# Start of the program
# =============================================================================
title('Quantum Monte Carlo')
section('Initialization')

# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print('Debugging:')
    all_debug()

# -----------------------------------------------------------------------------
# Read Input
# -----------------------------------------------------------------------------
print('Reading input...\n')
calculation_type, a, nmc, mc_trials, dt_VMC, dt_metro, tau, eref = read_input('INPUT')
title, charge, nxyz = read_xyz('XYZ')
print(nxyz)

# -----------------------------------------------------------------------------
# Output file
# -----------------------------------------------------------------------------
output_filename = 'OUTPUT'

# -----------------------------------------------------------------------------
# Print Input
# -----------------------------------------------------------------------------
print_input(title, charge, nxyz)

#-----------------------------------------------------------------------------
section('MC')
a = 1.2
ne = 1
R = (0., 0., 0.)
Z = [1]
nmc = 1000
dt_VMC = 5.
dt_metro = 1.
tau = 100
eref = -0.5
dt_PDMC = 0.05
mc_trials = 30
#-----------------------------------------------------------------------------
sVMC, syMe, geMe = VMC(a, ne, R, Z, mc_trials, nmc, dt_VMC, dt_metro)
output_VMC(sVMC, syMe, geMe)
sPDMC = PDMC(a, ne, R, Z, mc_trials, nmc, dt_PDMC, tau, eref)
output_PDMC(sPDMC)
# -----------------------------------------------------------------------------
end()
